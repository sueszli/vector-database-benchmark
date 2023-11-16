import ray
from ray.rllib.examples.env.simple_corridor import SimpleCorridor

class GPURequiringEnv(SimpleCorridor):
    """A dummy env that requires a GPU in order to work.

    The env here is a simple corridor env that additionally simulates a GPU
    check in its constructor via `ray.get_gpu_ids()`. If this returns an
    empty list, we raise an error.

    To make this env work, use `num_gpus_per_worker > 0` (RolloutWorkers
    requesting this many GPUs each) and - maybe - `num_gpus > 0` in case
    your local worker/driver must have an env as well. However, this is
    only the case if `create_env_on_driver`=True (default is False).
    """

    def __init__(self, config=None):
        if False:
            return 10
        super().__init__(config)
        gpus_available = ray.get_gpu_ids()
        assert len(gpus_available) > 0, 'Not enough GPUs for this env!'
        print('Env can see these GPUs: {}'.format(gpus_available))