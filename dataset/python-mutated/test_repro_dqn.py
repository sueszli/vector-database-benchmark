import unittest
import os
import ray
from ray.tune import register_env
import ray.rllib.algorithms.dqn as dqn
from ray.rllib.examples.env.deterministic_envs import create_cartpole_deterministic
from ray.rllib.utils.test_utils import check_reproducibilty

class TestReproDQN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        ray.shutdown()

    def test_reproducibility_dqn_cartpole(self):
        if False:
            i = 10
            return i + 15
        'Tests whether the algorithm is reproducible within 3 iterations\n        on discrete env cartpole.'
        register_env('DeterministicCartPole-v1', create_cartpole_deterministic)
        config = dqn.DQNConfig().environment(env='DeterministicCartPole-v1', env_config={'seed': 42})
        frameworks = ['torch']
        if int(os.environ.get('RLLIB_NUM_GPUS', 0)) == 0:
            frameworks.append('tf')
        check_reproducibilty(algo_class=dqn.DQN, algo_config=config, fw_kwargs={'frameworks': frameworks}, training_iteration=3)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))