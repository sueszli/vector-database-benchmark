from gymnasium.spaces import Space
from typing import Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from ray.rllib.utils.schedules import ConstantSchedule

@PublicAPI
class PerWorkerEpsilonGreedy(EpsilonGreedy):
    """A per-worker epsilon-greedy class for distributed algorithms.

    Sets the epsilon schedules of individual workers to a constant:
    0.4 ^ (1 + [worker-index] / float([num-workers] - 1) * 7)
    See Ape-X paper.
    """

    def __init__(self, action_space: Space, *, framework: str, num_workers: Optional[int], worker_index: Optional[int], **kwargs):
        if False:
            return 10
        'Create a PerWorkerEpsilonGreedy exploration class.\n\n        Args:\n            action_space: The gym action space used by the environment.\n            num_workers: The overall number of workers used.\n            worker_index: The index of the Worker using this\n                Exploration.\n            framework: One of None, "tf", "torch".\n        '
        epsilon_schedule = None
        assert worker_index <= num_workers, (worker_index, num_workers)
        if num_workers > 0:
            if worker_index > 0:
                (alpha, eps, i) = (7, 0.4, worker_index - 1)
                num_workers_minus_1 = float(num_workers - 1) if num_workers > 1 else 1.0
                constant_eps = eps ** (1 + i / num_workers_minus_1 * alpha)
                epsilon_schedule = ConstantSchedule(constant_eps, framework=framework)
            else:
                epsilon_schedule = ConstantSchedule(0.0, framework=framework)
        super().__init__(action_space, epsilon_schedule=epsilon_schedule, framework=framework, num_workers=num_workers, worker_index=worker_index, **kwargs)