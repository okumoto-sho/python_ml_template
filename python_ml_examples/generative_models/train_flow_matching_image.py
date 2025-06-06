import torch
import torchvision

from python_ml_examples.generative_models.flow_matching.velocity_models import Unet2D
from python_ml_examples.generative_models.flow_matching.flow_matching import (
    FlowMatching,
)
from python_ml_examples.generative_models.flow_matching.schedulers import (
    LinearScheduler,
)

from absl import flags, app
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoints_dir", "./checkpoints", "Directory to save checkpoints."
)
flags.DEFINE_string(
    "tensorboard_log_dir", "./runs", "Directory to save TensorBoard logs."
)
flags.DEFINE_string(
    "initial_checkpoint_path", None, "Path to the initial checkpoint to load."
)
flags.DEFINE_string("dataset_root_dir", "./dataset", "Root directory for the dataset.")
flags.DEFINE_integer("batch_size", 256, "Batch size for training.")
flags.DEFINE_integer("checkpoint_save_interval", 500, "Interval to save checkpoints.")
flags.DEFINE_integer("num_epochs", 500, "Number of epochs to train the model.")
flags.DEFINE_integer(
    "visualization_interval", 500, "Interval for visualizing generated images."
)
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for the optimizer.")


def main(_):
    # prepare dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
        ]
    )
    dataset = CIFAR10(root=FLAGS.dataset_root_dir, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)

    # prepare velocity model, scheduler, and flow matching instance
    velocity_model = Unet2D(
        in_channels=3,
        out_channels=3,
        down_block_types=[
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ],
        up_block_types=[
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ],
    ).cuda()
    scheduler = LinearScheduler()
    flow_matching = FlowMatching(
        model=velocity_model, scheduler=scheduler, learning_rate=FLAGS.learning_rate
    )

    if FLAGS.initial_checkpoint_path:
        flow_matching.load_state_dict(
            torch.load(FLAGS.initial_checkpoint_path, weights_only=True)
        )
        print(f"Loaded initial checkpoint from {FLAGS.initial_checkpoint_path}")

    # setup TensorBoard
    writer = SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)

    # train the model
    for epoch in range(FLAGS.num_epochs):
        for data in data_loader:
            x_1 = data[0].cuda()
            x_0 = torch.randn_like(x_1).cuda()
            loss, current_steps = flow_matching.train_one_step(x_0, x_1)
            print(
                f"Epoch [{epoch + 1}/{FLAGS.num_epochs}], Loss: {loss:.4f}, Steps: {current_steps}"
            )

            if current_steps % FLAGS.visualization_interval == 0:
                with torch.no_grad():
                    sampled_images = (
                        flow_matching.sample_from_source(
                            x_0[:4], t=1.0, num_ode_steps=32
                        )
                        .detach()
                        .cpu()
                    )
                    grid = torchvision.utils.make_grid(sampled_images, nrow=4)
                    writer.add_image("Generated Images", grid, current_steps)
                    writer.add_scalar("Train Loss", loss, current_steps)

            if current_steps % FLAGS.checkpoint_save_interval == 0:
                torch.save(
                    flow_matching.state_dict(),
                    f"{FLAGS.checkpoints_dir}/flow_matching_checkpoint_epoch_{epoch + 1}_step_{current_steps}.pth",
                )
                print(f"Checkpoint saved at epoch {epoch + 1}, step {current_steps}.")


if __name__ == "__main__":
    app.run(main)
