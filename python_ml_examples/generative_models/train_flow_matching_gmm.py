import torch


from python_ml_examples.generative_models.flow_matching.velocity_models import Mlp
from python_ml_examples.generative_models.flow_matching.schedulers import (
    CosineScheduler,
)
from python_ml_examples.generative_models.flow_matching.flow_matching import (
    FlowMatching,
)
from python_ml_examples.generative_models.flow_matching.utils import (
    save_2d_flow_matching_mp4,
    generate_gaussian_mixture,
)


n_batch_samples = 4096
dim = 2

# Create a fullt connected neural network model for flow matching
model = Mlp(n_dim=dim).cuda()

# Choose one of the schedulers to use
scheduler = CosineScheduler()

# Create flow matching model and train it on the moon dataset
flow_matching = FlowMatching(
    model=model,
    scheduler=scheduler,
)
x_1 = (
    torch.tensor(
        generate_gaussian_mixture(
            n_samples=n_batch_samples,
            means=[
                [0.0, 0.0],
                [2.0, 2.0],
                [-2.0, -2.0],
                [0.0, 5.0],
                [5.0, 0.0],
                [-10.0, 0.0],
                [0.0, -10.0],
            ],
            variances=[
                [0.4, 0.4],
                [0.4, 0.4],
                [0.4, 0.4],
                [0.1, 0.1],
                [0.1, 0.1],
                [0.1, 0.1],
                [0.1, 0.1],
            ],
        )
    )
    .float()
    .cuda()
)
for step in range(10000):
    x_0 = torch.randn_like(x_1).cuda()
    loss, current_steps = flow_matching.train_one_step(x_0, x_1)
    print(f"Loss: {loss:.4f}, Current Steps: {current_steps}")

x_0 = torch.randn(n_batch_samples, dim).cuda()
path = flow_matching.sample_path_from_source(x_0, num_ode_steps=100)

# Save generated path as mp4 file
paths = {}
paths["LinearScheduler"] = path
save_2d_flow_matching_mp4(paths, filename="flow_matching_moon.mp4")
