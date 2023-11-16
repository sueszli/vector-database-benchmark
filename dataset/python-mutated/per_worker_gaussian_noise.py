from gymnasium.spaces import Space
from typing import Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.exploration.gaussian_noise import GaussianNoise
from ray.rllib.utils.schedules import ConstantSchedule

@PublicAPI
class PerWorkerGaussianNoise(GaussianNoise):
    """A per-worker Gaussian noise class for distributed algorithms.

    Sets the `scale` schedules of individual workers to a constant:
    0.4 ^ (1 + [worker-index] / float([num-workers] - 1) * 7)
    See Ape-X paper.
    """

    def __init__(self, action_space: Space, *, framework: Optional[str], num_workers: Optional[int], worker_index: Optional[int], **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            action_space: The gym action space used by the environment.\n            num_workers: The overall number of workers used.\n            worker_index: The index of the Worker using this\n                Exploration.\n            framework: One of None, "tf", "torch".\n        '
        scale_schedule = None
        if num_workers > 0:
            if worker_index > 0:
                num_workers_minus_1 = float(num_workers - 1) if num_workers > 1 else 1.0
                exponent = 1 + worker_index / num_workers_minus_1 * 7
                scale_schedule = ConstantSchedule(0.4 ** exponent, framework=framework)
            else:
                scale_schedule = ConstantSchedule(0.0, framework=framework)
        super().__init__(action_space, scale_schedule=scale_schedule, framework=framework, **kwargs)