import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def generate_flow_matching_mp4(flow_data_dict, filename="flow_matching.mp4"):
    """
    Generates an MP4 video combining multiple flow dynamics.

    flow_data_dict: Dictionary where keys are scheduler names and values are Tensors,
                    each Tensor has shape (n_steps, batch_size, point_dim),
                    containing the points at each step for different flows.
    filename: Name of the output MP4 file
    """
    scheduler_names = list(flow_data_dict.keys())
    flow_data = list(flow_data_dict.values())
    n_flows = len(flow_data)
    n_cols = min(n_flows, 2)  # Limit the number of columns to 2
    n_rows = (n_flows + n_cols - 1) // n_cols  # Calculate rows based on columns
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), dpi=100
    )  # Set resolution
    axes = axes.flatten() if n_flows > 1 else [axes]  # Flatten axes for easy iteration

    def update(frame):
        for i, ax in enumerate(axes[:n_flows]):  # Only update axes with flows
            ax.clear()
            ax.set_title(f"{scheduler_names[i]} Dynamics at step {frame}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.axis("equal")
            ax.scatter(
                flow_data[i][frame, :, 0].detach().cpu().numpy(),
                flow_data[i][frame, :, 1].detach().cpu().numpy(),
                s=10,
                alpha=0.7,
            )

    ani = animation.FuncAnimation(
        fig, update, frames=flow_data[0].shape[0], interval=100
    )
    ani.save(filename, writer="ffmpeg")
    plt.close(fig)


def generate_gaussian_mixture(n_samples: int, means, variances):
    """
    Generate a 2D Gaussian Mixture Model.

    Args:
        n_samples (int): Total number of samples to generate.
        means (list of tuples): List of mean values for each cluster (e.g., [(x1, y1), (x2, y2), ...]).
        variances (list of floats): List of variances for each cluster.

    Returns:
        torch.Tensor: Generated samples as a tensor of shape (n_samples, 2).
    """
    n_clusters = len(means)
    assert (
        len(variances) == n_clusters
    ), "Number of variances must match the number of clusters."

    samples = []
    samples_per_cluster = n_samples // n_clusters

    for i in range(n_clusters):
        cluster_samples = torch.Tensor(means[i]) + torch.sqrt(
            torch.tensor(variances[i])
        ) * torch.randn(samples_per_cluster, 2)
        samples.append(cluster_samples)

    return torch.cat(samples, dim=0)
