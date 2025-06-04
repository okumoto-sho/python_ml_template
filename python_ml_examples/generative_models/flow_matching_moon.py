import torch

from sklearn.datasets import make_moons
from python_ml_examples.generative_models.flow_matching.velocity_models import Mlp
from python_ml_examples.generative_models.flow_matching.schedulers import (
    LinearScheduler,
    CosineScheduler,
    LinearVariancePreservingScheduler,
    PolynomialConvexScheduler,
)
from python_ml_examples.generative_models.flow_matching.flow_matching import (
    FlowMatching,
)
from python_ml_examples.generative_models.flow_matching.utils import (
    generate_flow_matching_mp4,
)


n_batch_samples = 512
dim = 2
model = Mlp(n_dim=dim).cuda()
scheduler_candidates = [
    LinearScheduler(),
    CosineScheduler(),
    LinearVariancePreservingScheduler(),
    PolynomialConvexScheduler(n=2),
]
paths = {}
for scheduler in scheduler_candidates:
    flow_matching = FlowMatching(
        model=model,
        scheduler=scheduler,
    )
    x_1 = (
        torch.tensor(make_moons(n_samples=n_batch_samples, noise=0.05)[0])
        .float()
        .cuda()
    )
    flow_matching.train(x_1, parameter_updating_steps=10000)
    path = flow_matching.sample_path(200, num_ode_steps=100)
    paths[scheduler.__class__.__name__] = path

generate_flow_matching_mp4(paths, filename="flow_matching_moon.mp4")
