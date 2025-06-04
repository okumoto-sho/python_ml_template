import torch


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
