import torch


from python_ml_examples.generative_models.flow_matching.velocity_models import Mlp
from python_ml_examples.generative_models.flow_matching.schedulers import (
    LinearScheduler,
)
from python_ml_examples.generative_models.flow_matching.flow_matching import (
    FlowMatching,
)
from python_ml_examples.generative_models.flow_matching.utils import (
    generate_flow_matching_mp4,
    generate_gaussian_mixture,
)


n_batch_samples = 4096
dim = 2

# Create a fullt connected neural network model for flow matching
model = Mlp(n_dim=dim).cuda()

# Choose one of the schedulers to use
scheduler = LinearScheduler()

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
flow_matching.train(x_1, parameter_updating_steps=10000)
path = flow_matching.sample_path(500, num_ode_steps=100)

# Save generated path as mp4 file
paths = {}
paths["LinearScheduler"] = path
generate_flow_matching_mp4(paths, filename="flow_matching_moon.mp4")
