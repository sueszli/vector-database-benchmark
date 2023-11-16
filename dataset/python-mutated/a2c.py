from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.a3c.a3c import A3CConfig, A3C
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING

class A2CConfig(A3CConfig):

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initializes a A2CConfig instance.'
        super().__init__(algo_class=A2C)
        self.microbatch_size = None
        self.num_rollout_workers = 2
        self.rollout_fragment_length = 'auto'
        self.sample_async = False
        self.min_time_s_per_iteration = 10

    @override(A3CConfig)
    def training(self, *, microbatch_size: Optional[int]=NotProvided, **kwargs) -> 'A2CConfig':
        if False:
            for i in range(10):
                print('nop')
        super().training(**kwargs)
        if microbatch_size is not NotProvided:
            self.microbatch_size = microbatch_size
        return self

@Deprecated(old='rllib/algorithms/a2c/', new='rllib_contrib/a2c/', help=ALGO_DEPRECATION_WARNING, error=True)
class A2C(A3C):

    @classmethod
    @override(A3C)
    def get_default_config(cls) -> AlgorithmConfig:
        if False:
            print('Hello World!')
        return A2CConfig()