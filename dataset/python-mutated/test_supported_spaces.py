import logging
import unittest
import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.test_utils import check_supported_spaces
logger = logging.getLogger(__name__)

class TestSupportedSpacesIMPALA(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            while True:
                i = 10
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_impala(self):
        if False:
            i = 10
            return i + 15
        check_supported_spaces('IMPALA', ImpalaConfig().resources(num_gpus=0).training(model={'fcnet_hiddens': [10]}))

class TestSupportedSpacesAPPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            while True:
                i = 10
        ray.shutdown()

    def test_appo(self):
        if False:
            return 10
        config = APPOConfig().resources(num_gpus=0).training(vtrace=False, model={'fcnet_hiddens': [10]})
        config.training(vtrace=True)
        check_supported_spaces('APPO', config)

class TestSupportedSpacesA3C(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

class TestSupportedSpacesPPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            while True:
                i = 10
        ray.shutdown()

    def test_ppo(self):
        if False:
            return 10
        config = PPOConfig().rollouts(num_rollout_workers=2, rollout_fragment_length=50).training(train_batch_size=100, num_sgd_iter=1, sgd_minibatch_size=50, model={'fcnet_hiddens': [10]})
        check_supported_spaces('PPO', config, check_bounds=True)

class TestSupportedSpacesPPONoPreprocessorGPU(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        ray.init(num_gpus=1)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            while True:
                i = 10
        ray.shutdown()

    def test_ppo_no_preprocessors_gpu(self):
        if False:
            return 10
        config = PPOConfig().rollouts(num_rollout_workers=2, rollout_fragment_length=50).training(train_batch_size=100, num_sgd_iter=1, sgd_minibatch_size=50, model={'fcnet_hiddens': [10]}).experimental(_disable_preprocessor_api=True).resources(num_gpus=1)
        check_supported_spaces('PPO', config, check_bounds=True, frameworks=['torch', 'tf'], use_gpu=True)

class TestSupportedSpacesDQN(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

    def test_dqn(self):
        if False:
            return 10
        config = DQNConfig().reporting(min_sample_timesteps_per_iteration=1).training(replay_buffer_config={'capacity': 1000})
        check_supported_spaces('DQN', config, frameworks=['tf2', 'torch', 'tf'])

class TestSupportedSpacesOffPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        ray.init(num_cpus=4)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_sac(self):
        if False:
            print('Hello World!')
        check_supported_spaces('SAC', SACConfig().training(replay_buffer_config={'capacity': 1000}), check_bounds=True)
if __name__ == '__main__':
    import pytest
    import sys
    class_ = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(pytest.main(['-v', __file__ + ('' if class_ is None else '::' + class_)]))