import torch
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons


def plot_2d_data(data, title="2D Data"):
    """
    Plots 2D data points.

    data: Tensor of shape (n_samples, 2)
    title: Title of the plot
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.axis("equal")
    plt.show()


class Mlp(torch.nn.Module):
    def __init__(self, n_dim: int, n_hidden: int = 64):
        super().__init__()
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


n_batch_samples = 512
dim = 2
model = Mlp(n_dim=dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
n_train_steps = 10000
loss_fn = torch.nn.MSELoss()

for step in range(n_train_steps):
    x_1 = (
        torch.tensor(make_moons(n_samples=n_batch_samples, noise=0.05)[0])
        .float()
        .cuda()
    )
    x_0 = torch.randn_like(x_1).cuda()
    t = torch.rand(n_batch_samples, 1).cuda()
    x_t = t * x_1 + (1 - t) * x_0
    loss = loss_fn(model(x_t, t), x_1 - x_0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {step}/{n_train_steps}, Loss: {loss.item():.4f}")

n_steps = 32
x_0 = torch.randn(n_batch_samples, dim).cuda()
x_t = x_0
cur_t = torch.zeros(n_batch_samples).cuda()
for _ in range(n_steps):
    h = 1 / n_steps
    x_t = x_t + h * model(x_t, cur_t)
    cur_t = cur_t + h
    print(f"Current time: {cur_t[0].item()}")
    plot_2d_data(
        x_t.detach().cpu(),
        title=f"Generated Data after Flow Matching at {cur_t[0].item():.2f}",
    )
