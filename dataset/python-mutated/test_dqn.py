from copy import deepcopy
import numpy as np
import unittest
import ray
import ray.rllib.algorithms.dqn as dqn
from ray.rllib.utils.test_utils import check, check_compute_single_action, check_train_results, framework_iterator

class TestDQN(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        ray.shutdown()

    def test_dqn_compilation(self):
        if False:
            i = 10
            return i + 15
        'Test whether DQN can be built on all frameworks.'
        num_iterations = 1
        config = dqn.dqn.DQNConfig().environment('CartPole-v1').rollouts(num_rollout_workers=2).training(num_steps_sampled_before_learning_starts=0)
        for _ in framework_iterator(config):
            print('Double-dueling')
            algo = config.build()
            for i in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)
            algo.stop()
            print('Rainbow')
            rainbow_config = deepcopy(config).training(num_atoms=10, noisy=True, double_q=True, dueling=True, n_step=5)
            algo = rainbow_config.build()
            for i in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)
            algo.stop()

    def test_dqn_compilation_integer_rewards(self):
        if False:
            for i in range(10):
                print('nop')
        'Test whether DQN can be built on all frameworks.\n        Unlike the previous test, this uses an environment with integer rewards\n        in order to test that type conversions are working correctly.'
        num_iterations = 1
        config = dqn.dqn.DQNConfig().environment('Taxi-v3').rollouts(num_rollout_workers=2).training(num_steps_sampled_before_learning_starts=0)
        for _ in framework_iterator(config):
            print('Double-dueling')
            algo = config.build()
            for i in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)
            algo.stop()
            print('Rainbow')
            rainbow_config = deepcopy(config).training(num_atoms=10, noisy=True, double_q=True, dueling=True, n_step=5)
            algo = rainbow_config.build()
            for i in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)
            algo.stop()

    def test_dqn_exploration_and_soft_q_config(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests, whether a DQN Agent outputs exploration/softmaxed actions.'
        config = dqn.dqn.DQNConfig().environment('FrozenLake-v1').rollouts(num_rollout_workers=0).environment(env_config={'is_slippery': False, 'map_name': '4x4'}).training(num_steps_sampled_before_learning_starts=0)
        obs = np.array(0)
        for _ in framework_iterator(config):
            algo = config.build()
            a_ = algo.compute_single_action(obs, explore=False)
            for _ in range(50):
                a = algo.compute_single_action(obs, explore=False)
                check(a, a_)
            actions = []
            for _ in range(50):
                actions.append(algo.compute_single_action(obs))
            check(np.std(actions), 0.0, false=True)
            algo.stop()
            config.exploration(exploration_config={'type': 'SoftQ', 'temperature': 1e-06})
            algo = config.build()
            actions = [algo.compute_single_action(obs)]
            for _ in range(50):
                actions.append(algo.compute_single_action(obs))
            check(np.std(actions), 0.0, decimals=3)
            algo.stop()
            config.exploration_config['temperature'] = 1.0
            algo = config.build()
            a_ = algo.compute_single_action(obs, explore=False)
            for _ in range(50):
                a = algo.compute_single_action(obs, explore=False)
                check(a, a_)
            actions = []
            for _ in range(300):
                actions.append(algo.compute_single_action(obs))
            check(np.std(actions), 0.0, false=True)
            algo.stop()
            config.exploration(exploration_config={'type': 'Random'}, explore=True)
            algo = config.build()
            actions = []
            for _ in range(300):
                actions.append(algo.compute_single_action(obs))
            check(np.std(actions), 0.0, false=True)
            algo.stop()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))