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
from torchvision.datasets import MNIST
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


def main(_):
    # prepare dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            transforms.Resize((32, 32)),
        ]
    )
    dataset = MNIST(root="./dataset", download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # prepare velocity model, scheduler, and flow matching instance
    velocity_model = Unet2D(in_channels=1, out_channels=1).cuda()
    scheduler = LinearScheduler()
    flow_matching = FlowMatching(
        model=velocity_model, scheduler=scheduler, learning_rate=1e-5
    )

    # set training hyper parameters
    num_epochs = 5
    visualization_interval = 500
    checkpoint_interval = 200

    # setup TensorBoard
    writer = SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)

    # train the model
    for epoch in range(num_epochs):
        for data in data_loader:
            x_1 = data[0].cuda()
            x_0 = torch.randn_like(x_1).cuda()
            loss, current_steps = flow_matching.train_one_step(x_0, x_1)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Steps: {current_steps}"
            )

            if current_steps % visualization_interval == 0:
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
            if current_steps % checkpoint_interval == 0:
                torch.save(
                    velocity_model.state_dict(),
                    f"{FLAGS.checkpoints_dir}/flow_matching_checkpoint_epoch_{epoch + 1}_step_{current_steps}.pth",
                )
                print(f"Checkpoint saved at epoch {epoch + 1}, step {current_steps}.")


if __name__ == "__main__":
    app.run(main)
