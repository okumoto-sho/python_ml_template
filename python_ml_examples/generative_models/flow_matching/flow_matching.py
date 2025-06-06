import torch
import torch.nn as nn

from typing import NamedTuple
from python_ml_examples.generative_models.flow_matching.velocity_models import (
    FlowVelocityModel,
)
from python_ml_examples.generative_models.flow_matching.schedulers import Scheduler


class TrainStats(NamedTuple):
    """Statistics for training."""

    loss: float
    current_train_steps: int


class FlowMatching(nn.Module):
    def __init__(
        self,
        model: FlowVelocityModel,
        scheduler: Scheduler,
        optimizer: str = "Adam",
        learning_rate: float = 1e-2,
        loss_fn: str = "MSELoss",
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer)(
            model.parameters(), lr=learning_rate
        )
        self.loss_fn = getattr(torch.nn, loss_fn)()
        self.device = device
        self.current_train_steps = 0

    def train_one_step(self, x_0: torch.Tensor, x_1: torch.Tensor) -> TrainStats:
        n_batch_samples = x_0.shape[0]
        t = torch.rand(n_batch_samples).to(device=self.device)
        x_t, dx_t = self.scheduler.sample(x_0, x_1, t)
        loss = self.loss_fn(self.model(x_t, t), dx_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.current_train_steps += 1
        return TrainStats(
            loss=loss.item(),
            current_train_steps=self.current_train_steps,
        )

    def sample_from_source(
        self, x_0: torch.Tensor, t: float = 1.0, num_ode_steps: int = 32
    ) -> torch.Tensor:
        step_width = t / num_ode_steps
        n_samples = x_0.shape[0]
        x_t = x_0
        cur_t = 0.0
        for step in range(num_ode_steps):
            times = torch.tensor([cur_t] * n_samples).to(device=self.device)
            x_t = x_t + step_width * self.model(
                x_t,
                times,
            )
            cur_t += step_width
        return x_t

    def sample_path_from_source(
        self, x_0: torch.Tensor, t: float = 1.0, num_ode_steps=32
    ) -> torch.Tensor:
        step_width = t / num_ode_steps
        path = torch.zeros((num_ode_steps, *x_0.shape)).to(device=self.device)
        x_t = x_0
        n_samples = x_0.shape[0]
        path[0, :] = x_t
        cur_t = 0.0
        for step in range(num_ode_steps):
            times = torch.tensor([cur_t] * n_samples).to(device=self.device)
            x_t = x_t + step_width * self.model(
                x_t,
                times,
            )
            path[step, :] = x_t
            cur_t += step_width
        return path
