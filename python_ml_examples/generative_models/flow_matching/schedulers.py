import torch
import abc

from numpy import pi


class Scheduler(abc.ABC, torch.nn.Module):
    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        """
        Sample a point from the scheduler.
        x_0: Tensor of shape (batch_size, n_dim)
        x_1: Tensor of shape (batch_size, n_dim)
        t: Tensor of shape (batch_size,) representing time
        Returns:
            A tuple (x_t, dx_t) where:
            x_t: Sampled tensor of shape (batch_size, n_dim)
            dx_t: Derivative of the sampled tensor with respect to time
        """
        raise NotImplementedError("Subclasses should implement the sample method.")


class LinearScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        alpha_t = t.view(-1, *[1] * (x_0.ndim - 1))
        sigma_t = (1 - t).view(-1, *[1] * (x_0.ndim - 1))
        d_alpha_t = torch.ones_like(alpha_t)
        d_sigma_t = -torch.ones_like(sigma_t)
        return sigma_t * x_0 + alpha_t * x_1, d_sigma_t * x_0 + d_alpha_t * x_1


class CosineScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        alpha_t = torch.sin(pi / 2 * t).view(-1, *[1] * (x_0.ndim - 1))
        sigma_t = torch.cos(pi / 2 * t).view(-1, *[1] * (x_0.ndim - 1))
        d_alpha_t = (pi / 2 * torch.cos(pi / 2 * t)).view(-1, *[1] * (x_0.ndim - 1))
        d_sigma_t = (-pi / 2 * torch.sin(pi / 2 * t)).view(-1, *[1] * (x_0.ndim - 1))
        return sigma_t * x_0 + alpha_t * x_1, d_sigma_t * x_0 + d_alpha_t * x_1


class LinearVariancePreservingScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        alpha_t = t.view(-1, *[1] * (x_0.ndim - 1))
        sigma_t = torch.sqrt(1 - t**2).view(-1, *[1] * (x_0.ndim - 1))
        d_alpha_t = torch.ones_like(alpha_t).view(-1, *[1] * (x_0.ndim - 1))
        d_sigma_t = (-t / (torch.sqrt(1 - t**2) + 1e-8)).view(-1, *[1] * (x_0.ndim - 1))
        return sigma_t * x_0 + alpha_t * x_1, d_sigma_t * x_0 + d_alpha_t * x_1


class PolynomialConvexScheduler(Scheduler):
    def __init__(self, n: int = 2):
        super().__init__()
        self.n = n

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        alpha_t = (t**self.n).view(-1, *[1] * (x_0.ndim - 1))
        sigma_t = (1 - t**self.n).view(-1, *[1] * (x_0.ndim - 1))
        d_alpha_t = (self.n * t ** (self.n - 1)).view(-1, *[1] * (x_0.ndim - 1))
        d_sigma_t = -(self.n * t ** (self.n - 1)).view(-1, *[1] * (x_0.ndim - 1))
        return sigma_t * x_0 + alpha_t * x_1, d_sigma_t * x_0 + d_alpha_t * x_1
