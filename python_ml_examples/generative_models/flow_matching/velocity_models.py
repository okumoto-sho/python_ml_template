import torch

from diffusers.models.unets import UNet2DModel


class FlowVelocityModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError("Subclasses should implement the forward method.")

    @property
    def n_dim(self):
        raise NotImplementedError("Subclasses should implement the n_dim property.")


class Mlp(FlowVelocityModel):
    def __init__(self, n_dim: int, n_hidden: int = 64):
        super().__init__()
        self._n_dim = n_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_dim + 1, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: Input tensor of shape (batch_size, n_dim)
        t: Time tensor of shape (batch_size,)
        """
        x_in = torch.cat([x, t.view(-1, 1)], dim=1)
        return self.net(x_in)

    @property
    def n_dim(self):
        return self._n_dim


class Unet2D(FlowVelocityModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(64, 128, 256, 512),
    ):
        super().__init__()
        self.unet = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        unet_output = self.unet(sample=x, timestep=t)
        return unet_output.sample
