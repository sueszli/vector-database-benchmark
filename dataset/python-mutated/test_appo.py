import unittest
import ray
import ray.rllib.algorithms.appo as appo
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.rllib.utils.test_utils import check_compute_single_action, check_train_results, framework_iterator

class TestAPPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        ray.shutdown()

    def test_appo_compilation(self):
        if False:
            while True:
                i = 10
        'Test whether APPO can be built with both frameworks.'
        config = appo.APPOConfig().rollouts(num_rollout_workers=1)
        num_iterations = 2
        for _ in framework_iterator(config):
            print('w/o v-trace')
            config.vtrace = False
            algo = config.build(env='CartPole-v1')
            for i in range(num_iterations):
                results = algo.train()
                print(results)
                check_train_results(results)
            check_compute_single_action(algo)
            algo.stop()
            print('w/ v-trace')
            config.vtrace = True
            algo = config.build(env='CartPole-v1')
            for i in range(num_iterations):
                results = algo.train()
                print(results)
                check_train_results(results)
            check_compute_single_action(algo)
            algo.stop()

    def test_appo_compilation_use_kl_loss(self):
        if False:
            return 10
        'Test whether APPO can be built with kl_loss enabled.'
        config = appo.APPOConfig().rollouts(num_rollout_workers=1).training(use_kl_loss=True)
        num_iterations = 2
        for _ in framework_iterator(config):
            algo = config.build(env='CartPole-v1')
            for i in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)
            algo.stop()

    def test_appo_two_optimizers_two_lrs(self):
        if False:
            for i in range(10):
                print('nop')
        config = appo.APPOConfig().rollouts(num_rollout_workers=1).training(_separate_vf_optimizer=True, _lr_vf=0.002, model={'vf_share_layers': False})
        num_iterations = 2
        for _ in framework_iterator(config, frameworks=('torch', 'tf2', 'tf')):
            algo = config.build(env='CartPole-v1')
            for i in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)
            algo.stop()

    def test_appo_entropy_coeff_schedule(self):
        if False:
            while True:
                i = 10
        config = appo.APPOConfig().rollouts(num_rollout_workers=1, batch_mode='truncate_episodes', rollout_fragment_length=10).resources(num_gpus=0).training(train_batch_size=20, entropy_coeff=0.01, entropy_coeff_schedule=[[0, 0.1], [100, 0.01], [300, 0.001], [500, 0.0001]]).reporting(min_train_timesteps_per_iteration=20, min_time_s_per_iteration=0)

        def _step_n_times(algo, n: int):
            if False:
                for i in range(10):
                    print('nop')
            'Step Algorithm n times.\n\n            Returns:\n                learning rate at the end of the execution.\n            '
            for _ in range(n):
                results = algo.train()
                print(algo.workers.local_worker().global_vars)
                print(results)
            return results['info'][LEARNER_INFO][DEFAULT_POLICY_ID][LEARNER_STATS_KEY]['entropy_coeff']
        for _ in framework_iterator(config, frameworks=('torch', 'tf')):
            algo = config.build(env='CartPole-v1')
            coeff = _step_n_times(algo, 10)
            self.assertLessEqual(coeff, 0.01)
            self.assertGreaterEqual(coeff, 0.001)
            coeff = _step_n_times(algo, 20)
            self.assertLessEqual(coeff, 0.001)
            algo.stop()

    def test_appo_learning_rate_schedule(self):
        if False:
            print('Hello World!')
        config = appo.APPOConfig().rollouts(num_rollout_workers=1, batch_mode='truncate_episodes', rollout_fragment_length=10).resources(num_gpus=0).training(train_batch_size=20, entropy_coeff=0.01, lr_schedule=[[0, 0.05], [500, 0.0]]).reporting(min_train_timesteps_per_iteration=20, min_time_s_per_iteration=0)

        def _step_n_times(algo, n: int):
            if False:
                for i in range(10):
                    print('nop')
            'Step Algorithm n times.\n\n            Returns:\n                learning rate at the end of the execution.\n            '
            for _ in range(n):
                results = algo.train()
                print(algo.workers.local_worker().global_vars)
                print(results)
            return results['info'][LEARNER_INFO][DEFAULT_POLICY_ID][LEARNER_STATS_KEY]['cur_lr']
        for _ in framework_iterator(config):
            algo = config.build(env='CartPole-v1')
            lr1 = _step_n_times(algo, 10)
            lr2 = _step_n_times(algo, 10)
            self.assertGreater(lr1, lr2)
            algo.stop()

    def test_appo_model_variables(self):
        if False:
            for i in range(10):
                print('nop')
        config = appo.APPOConfig().rollouts(num_rollout_workers=1, batch_mode='truncate_episodes', rollout_fragment_length=10).resources(num_gpus=0).training(train_batch_size=20).training(model={'fcnet_hiddens': [16]})
        for _ in framework_iterator(config, frameworks=['tf2', 'torch']):
            algo = config.build(env='CartPole-v1')
            state = algo.get_policy(DEFAULT_POLICY_ID).get_state()
            self.assertEqual(len(state['weights']), 6)
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))