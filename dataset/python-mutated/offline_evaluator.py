import abc
import os
import logging
from typing import Dict, Any
from ray.data import Dataset
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.typing import SampleBatchType
logger = logging.getLogger(__name__)

@DeveloperAPI
class OfflineEvaluator(abc.ABC):
    """Interface for an offline evaluator of a policy"""

    @DeveloperAPI
    def __init__(self, policy: Policy, **kwargs):
        if False:
            while True:
                i = 10
        'Initializes an OffPolicyEstimator instance.\n\n        Args:\n            policy: Policy to evaluate.\n            kwargs: forward compatibility placeholder.\n        '
        self.policy = policy

    @abc.abstractmethod
    @DeveloperAPI
    def estimate(self, batch: SampleBatchType, **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Returns the evaluation results for the given batch of episodes.\n\n        Args:\n            batch: The batch to evaluate.\n            kwargs: forward compatibility placeholder.\n\n        Returns:\n            The evaluation done on the given batch. The returned\n            dict can be any arbitrary mapping of strings to metrics.\n        '
        raise NotImplementedError

    @DeveloperAPI
    def train(self, batch: SampleBatchType, **kwargs) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Sometimes you need to train a model inside an evaluator. This method\n        abstracts the training process.\n\n        Args:\n            batch: SampleBatch to train on\n            kwargs: forward compatibility placeholder.\n\n        Returns:\n            Any optional metrics to return from the evaluator\n        '
        return {}

    @ExperimentalAPI
    def estimate_on_dataset(self, dataset: Dataset, *, n_parallelism: int=os.cpu_count()) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Calculates the estimate of the metrics based on the given offline dataset.\n\n        Typically, the dataset is passed through only once via n_parallel tasks in\n        mini-batches to improve the run-time of metric estimation.\n\n        Args:\n            dataset: The ray dataset object to do offline evaluation on.\n            n_parallelism: The number of parallelism to use for the computation.\n\n        Returns:\n            Dict[str, Any]: A dictionary of the estimated values.\n        '