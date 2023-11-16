from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ddpg.ddpg import DDPG, DDPGConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, Deprecated, ALGO_DEPRECATION_WARNING

class TD3Config(DDPGConfig):

    def __init__(self, algo_class=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(algo_class=algo_class or TD3)
        self.twin_q = True
        self.policy_delay = 2
        self.smooth_target_policy = (True,)
        self.l2_reg = 0.0
        self.tau = 0.005
        self.train_batch_size = 100
        self.replay_buffer_config = {'type': 'MultiAgentReplayBuffer', 'prioritized_replay': DEPRECATED_VALUE, 'capacity': 1000000, 'worker_side_prioritization': False}
        self.num_steps_sampled_before_learning_starts = 10000
        self.exploration_config = {'type': 'GaussianNoise', 'random_timesteps': 10000, 'stddev': 0.1, 'initial_scale': 1.0, 'final_scale': 1.0, 'scale_timesteps': 1}

@Deprecated(old='rllib/algorithms/td3/', new='rllib_contrib/td3/', help=ALGO_DEPRECATION_WARNING, error=True)
class TD3(DDPG):

    @classmethod
    @override(DDPG)
    def get_default_config(cls) -> AlgorithmConfig:
        if False:
            i = 10
            return i + 15
        return TD3Config()