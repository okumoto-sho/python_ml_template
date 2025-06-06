import torch

from python_ml_examples.generative_models.flow_matching.velocity_models import Unet2D
from python_ml_examples.generative_models.flow_matching.flow_matching import (
    FlowMatching,
)
from python_ml_examples.generative_models.flow_matching.schedulers import (
    LinearScheduler,
)
from python_ml_examples.generative_models.flow_matching.utils import (
    save_image_flow_matching_mp4,
)

from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_path", None, "Path to the trained flow matching model checkpoint."
)
flags.DEFINE_string(
    "video_path",
    "image_dynamics.mp4",
    "Path to save the generated video of image dynamics.",
)
flags.DEFINE_integer("batch_size", 32, "Batch size for sampling images.")


def main(_):
    velocity_model = Unet2D(in_channels=3, out_channels=3).cuda()
    scheduler = LinearScheduler()
    flow_matching = FlowMatching(
        model=velocity_model,
        scheduler=scheduler,
    )
    flow_matching.load_state_dict(torch.load(FLAGS.checkpoint_path, weights_only=True))

    with torch.no_grad():
        x_0 = torch.randn(FLAGS.batch_size, 3, 32, 32).cuda()
        path = flow_matching.sample_path_from_source(x_0=x_0, num_ode_steps=64)
        save_image_flow_matching_mp4(
            image_flow_path=path,
            save_path=FLAGS.video_path,
        )


if __name__ == "__main__":
    app.run(main)
