from typing import List, Optional, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING

class A3CConfig(AlgorithmConfig):

    def __init__(self, algo_class=None):
        if False:
            i = 10
            return i + 15
        'Initializes a A3CConfig instance.'
        super().__init__(algo_class=algo_class or A3C)
        self.use_critic = True
        self.use_gae = True
        self.lambda_ = 1.0
        self.grad_clip = 40.0
        self.grad_clip_by = 'global_norm'
        self.lr_schedule = None
        self.vf_loss_coeff = 0.5
        self.entropy_coeff = 0.01
        self.entropy_coeff_schedule = None
        self.sample_async = True
        self.num_rollout_workers = 2
        self.rollout_fragment_length = 10
        self.lr = 0.0001
        self.min_time_s_per_iteration = 5
        self.exploration_config = {'type': 'StochasticSampling'}

    @override(AlgorithmConfig)
    def training(self, *, lr_schedule: Optional[List[List[Union[int, float]]]]=NotProvided, use_critic: Optional[bool]=NotProvided, use_gae: Optional[bool]=NotProvided, lambda_: Optional[float]=NotProvided, grad_clip: Optional[float]=NotProvided, vf_loss_coeff: Optional[float]=NotProvided, entropy_coeff: Optional[float]=NotProvided, entropy_coeff_schedule: Optional[List[List[Union[int, float]]]]=NotProvided, sample_async: Optional[bool]=NotProvided, **kwargs) -> 'A3CConfig':
        if False:
            return 10
        super().training(**kwargs)
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if use_critic is not NotProvided:
            self.lr_schedule = use_critic
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if sample_async is not NotProvided:
            self.sample_async = sample_async
        return self

@Deprecated(old='rllib/algorithms/a3c/', new='rllib_contrib/a3c/', help=ALGO_DEPRECATION_WARNING, error=True)
class A3C(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        if False:
            return 10
        return A3CConfig()