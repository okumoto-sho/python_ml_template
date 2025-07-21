import dataclasses
import time
import torch
import math
import os


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn as nn
import sys
import math
import tiktoken

from dataclasses import dataclass
from torch.nn import functional as F


@dataclasses.dataclass
class DDPInfo:
    is_ddp: bool
    local_rank: int
    world_size: int
    is_master_process: bool
    device: torch.device


def create_logger():
    import logging
    import sys
    from colorama import Fore, Style

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            level_color = {
                "DEBUG": Fore.BLUE,
                "INFO": Fore.GREEN,
                "WARNING": Fore.YELLOW,
                "ERROR": Fore.RED,
                "CRITICAL": Fore.MAGENTA,
            }.get(record.levelname, Fore.WHITE)
            reset = Style.RESET_ALL
            record.msg = f"{level_color}{record.msg}{reset}"
            return super().format(record)

    logger = logging.getLogger("gpt2")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = create_logger()


def init_ddp():
    env = os.environ
    is_ddp = "RANK" in env and "WORLD_SIZE" in env
    if is_ddp:
        local_rank = int(env["LOCAL_RANK"])
        world_size = int(env["WORLD_SIZE"])
        is_master_process = local_rank == 0
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)
        logger.info(
            f"Initialized DDP with local_rank: {local_rank}, world_size: {world_size} on device: {device}"
        )
    else:
        local_rank = 0
        world_size = 1
        is_master_process = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Running in single process mode, not using DDP.")

    if is_ddp:
        dist.barrier()  # ensure all processes are synchronized before proceeding

    torch.cuda.set_device(local_rank)
    return DDPInfo(
        is_ddp=is_ddp,
        local_rank=local_rank,
        world_size=world_size,
        is_master_process=is_master_process,
        device=device,
    )


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # $ Batch size, sequence length, embedding size
        assert C == self.c_attn.in_features

        qkv: torch.Tensor = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, T, C) * 3
        # (B, n_head, T, C // n_head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = (
        #    att @ v
        # )  # (B, n_head, T, T), (B, n_head, T, C//n_head) --> (B, n_head, T, C//n_head)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing between the token embedding and the output head
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()  # Batch size, sequence length
        assert (
            T <= self.transformer.wpe.num_embeddings
        ), "Cannot forward sequence of length %d, block size is only %d" % (
            T,
            self.transformer.wpe.num_embeddings,
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embed)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embed)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1))
            return logits, loss
        return logits

    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
        ], f"Unknown model type: {model_type}"
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay: float, learning_rate: float):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(
            f"num decay params: {num_decay_params}, num no decay params: {num_no_decay_params}"
        )
        optimizer = torch.optim.AdamW(
            optim_groups, learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True
        )
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T, local_rank, world_size):
        self.B, self.T, self.local_rank, self.world_size = B, T, local_rank, world_size
        with open("./python_ml_examples/gpt-2/input.txt", "r") as f:
            self.text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        self.tokens = enc.encode(self.text)
        self.current_index = self.B * self.T * self.local_rank

    def next_batch(self):
        chunk_size = self.B * self.T
        batch_size = self.B * self.T * self.world_size
        if self.current_index + (self.local_rank * chunk_size + 1) >= len(self.tokens):
            self.current_index = self.B * self.T * self.local_rank

        start_idx = self.current_index
        end_idx = self.current_index + chunk_size
        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1 : end_idx + 1]

        x = torch.tensor(x, dtype=torch.long).view(self.B, self.T)
        y = torch.tensor(y, dtype=torch.long).view(self.B, self.T)

        self.current_index += batch_size
        return x, y


def cosine_scheduler_lr(
    steps: int,
    warmup_steps: int = 10,
    max_steps: int = 50,
    max_lr=6e-4,
    min_lr_ratio: float = 0.1,
):
    min_lr = min_lr_ratio * max_lr
    if steps < warmup_steps:
        return (1 + steps) / warmup_steps * max_lr
    if steps > max_steps:
        return min_lr

    coeff = 0.5 * (
        1 + math.cos(math.pi * (steps - warmup_steps) / (max_steps - warmup_steps))
    )
    return min_lr + coeff * (max_lr - min_lr)


ddp_info = init_ddp()
model = GPT(GPTConfig())
model.to("cuda")
model = torch.compile(model)
if ddp_info.is_ddp:
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[ddp_info.local_rank], output_device=ddp_info.local_rank
    )

total_batch_size = 25600  # in total number of tokens per step. B * T * grad_acum_steps
B = 5
T = 512
grad_acum_steps = total_batch_size // (B * T * ddp_info.world_size)
assert (
    total_batch_size % (B * T * ddp_info.world_size) == 0
), "total_batch_size must be divisible by B * T"
if ddp_info.is_master_process:
    logger.info(
        f"Using total batch size: {total_batch_size}, B: {B}, T: {T}, grad_acum_steps: {grad_acum_steps}"
    )

train_data_loader = DataLoaderLite(
    B=B, T=T, local_rank=ddp_info.local_rank, world_size=ddp_info.world_size
)
n_epochs = 50
optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4)
scaler = torch.GradScaler()
for i in range(n_epochs):
    t_0 = time.time()
    x, y = train_data_loader.next_batch()
    x = x.to("cuda")
    y = y.to("cuda")
    optimizer.zero_grad()

    for grad_acum_step in range(grad_acum_steps):
        with torch.autocast("cuda", torch.float16):
            logits, loss = model(x, y)
            loss = loss / grad_acum_steps
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    lr = cosine_scheduler_lr(i)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    t_1 = time.time()

    logger.info(
        f"step {i + 1}, loss: {loss.item():.4f}, time: {(t_1 - t_0) * 1000} tokens/sec: {B * T * grad_acum_steps / (t_1 - t_0):.2f}, lr: {lr} norm: {norm}"
    )

if ddp_info.is_ddp:
    dist.destroy_process_group()
sys.exit(0)

"""
torch.manual_seed(42)
torch.cuda.manual_seed(42)
max_length = 30
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_sizse)
        logits = logits[:, -1, :]  # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # (B, 50)
        next_token_idx = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
        xcol = torch.gather(topk_indices, dim=-1, index=next_token_idx)
        x = torch.cat((x, xcol), dim=1)
print("Generated tokens:", x)

for i in range(x.size(0)):
    tokens = x[i, :max_length].tolist()
    text = enc.decode(tokens)
    print(">", text)
"""
