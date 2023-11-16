from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING

class DDPPOConfig(PPOConfig):

    def __init__(self, algo_class=None):
        if False:
            while True:
                i = 10
        super().__init__(algo_class=algo_class or DDPPO)
        self.keep_local_weights_in_sync = True
        self.torch_distributed_backend = 'gloo'
        self.num_rollout_workers = 2
        self.num_envs_per_worker = 5
        self.sgd_minibatch_size = 50
        self.num_sgd_iter = 10
        self.framework_str = 'torch'
        self.num_gpus = 0
        self.num_gpus_per_worker = 1
        self.train_batch_size = 500
        self.kl_coeff = 0.0
        self.kl_target = 0.0
        self._enable_new_api_stack = False
        self.exploration_config = {'type': 'StochasticSampling'}

    @override(PPOConfig)
    def training(self, *, keep_local_weights_in_sync: Optional[bool]=NotProvided, torch_distributed_backend: Optional[str]=NotProvided, **kwargs) -> 'DDPPOConfig':
        if False:
            return 10
        super().training(**kwargs)
        if keep_local_weights_in_sync is not NotProvided:
            self.keep_local_weights_in_sync = keep_local_weights_in_sync
        if torch_distributed_backend is not NotProvided:
            self.torch_distributed_backend = torch_distributed_backend
        return self

@Deprecated(old='rllib/algorithms/ddppo/', new='rllib_contrib/ddppo/', help=ALGO_DEPRECATION_WARNING, error=True)
class DDPPO(PPO):

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        if False:
            i = 10
            return i + 15
        return DDPPOConfig()