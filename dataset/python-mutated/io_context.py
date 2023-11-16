import os
from typing import Optional, TYPE_CHECKING
from ray.rllib.utils.annotations import PublicAPI
if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.evaluation.sampler import SamplerInput
    from ray.rllib.evaluation.rollout_worker import RolloutWorker

@PublicAPI
class IOContext:
    """Class containing attributes to pass to input/output class constructors.

    RLlib auto-sets these attributes when constructing input/output classes,
    such as InputReaders and OutputWriters.
    """

    @PublicAPI
    def __init__(self, log_dir: Optional[str]=None, config: Optional['AlgorithmConfig']=None, worker_index: int=0, worker: Optional['RolloutWorker']=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a IOContext object.\n\n        Args:\n            log_dir: The logging directory to read from/write to.\n            config: The (main) AlgorithmConfig object.\n            worker_index: When there are multiple workers created, this\n                uniquely identifies the current worker. 0 for the local\n                worker, >0 for any of the remote workers.\n            worker: The RolloutWorker object reference.\n        '
        from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        self.log_dir = log_dir or os.getcwd()
        self.config = config or AlgorithmConfig().offline_data(actions_in_input_normalized=worker is None).training(train_batch_size=1)
        self.worker_index = worker_index
        self.worker = worker

    @PublicAPI
    def default_sampler_input(self) -> Optional['SamplerInput']:
        if False:
            return 10
        "Returns the RolloutWorker's SamplerInput object, if any.\n\n        Returns None if the RolloutWorker has no SamplerInput. Note that local\n        workers in case there are also one or more remote workers by default\n        do not create a SamplerInput object.\n\n        Returns:\n            The RolloutWorkers' SamplerInput object or None if none exists.\n        "
        return self.worker.sampler

    @property
    @PublicAPI
    def input_config(self):
        if False:
            while True:
                i = 10
        return self.config.get('input_config', {})

    @property
    @PublicAPI
    def output_config(self):
        if False:
            return 10
        return self.config.get('output_config', {})