import unittest
import numpy as np
import rllib_simple_q.simple_q.simple_q as simple_q
from rllib_simple_q.simple_q.simple_q_tf_policy import SimpleQTF2Policy
from rllib_simple_q.simple_q.simple_q_torch_policy import SimpleQTorchPolicy
import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import DEFAULT_POLICY_ID, LEARNER_INFO, LEARNER_STATS_KEY
from ray.rllib.utils.numpy import fc, huber_loss, one_hot
from ray.rllib.utils.test_utils import check, check_compute_single_action, check_train_results, framework_iterator
(tf1, tf, tfv) = try_import_tf()

class TestSimpleQ(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

    def test_simple_q_compilation(self):
        if False:
            for i in range(10):
                print('nop')
        'Test whether SimpleQ can be built on all frameworks.'
        config = simple_q.SimpleQConfig().rollouts(num_rollout_workers=0, compress_observations=True).training(num_steps_sampled_before_learning_starts=0)
        num_iterations = 2
        for _ in framework_iterator(config, with_eager_tracing=True):
            algo = config.build(env='CartPole-v1')
            rw = algo.workers.local_worker()
            for i in range(num_iterations):
                sb = rw.sample()
                assert sb.count == config.rollout_fragment_length
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)

    def test_simple_q_loss_function(self):
        if False:
            return 10
        'Tests the Simple-Q loss function results on all frameworks.'
        config = simple_q.SimpleQConfig().rollouts(num_rollout_workers=0)
        config.training(model={'fcnet_hiddens': [10], 'fcnet_activation': 'linear'}, num_steps_sampled_before_learning_starts=0).environment('CartPole-v1')
        for fw in framework_iterator(config):
            trainer = config.build()
            policy = trainer.get_policy()
            input_ = SampleBatch({SampleBatch.CUR_OBS: np.random.random(size=(2, 4)), SampleBatch.ACTIONS: np.array([0, 1]), SampleBatch.REWARDS: np.array([0.4, -1.23]), SampleBatch.TERMINATEDS: np.array([False, False]), SampleBatch.NEXT_OBS: np.random.random(size=(2, 4)), SampleBatch.EPS_ID: np.array([1234, 1234]), SampleBatch.AGENT_INDEX: np.array([0, 0]), SampleBatch.ACTION_LOGP: np.array([-0.1, -0.1]), SampleBatch.ACTION_DIST_INPUTS: np.array([[0.1, 0.2], [-0.1, -0.2]]), SampleBatch.ACTION_PROB: np.array([0.1, 0.2]), 'q_values': np.array([[0.1, 0.2], [0.2, 0.1]])})
            vars = policy.get_weights()
            if isinstance(vars, dict):
                vars = list(vars.values())
            vars_t = policy.target_model.variables()
            if fw == 'tf':
                vars_t = policy.get_session().run(vars_t)
            q_t = np.sum(one_hot(input_[SampleBatch.ACTIONS], 2) * fc(fc(input_[SampleBatch.CUR_OBS], vars[0 if fw != 'torch' else 2], vars[1 if fw != 'torch' else 3], framework=fw), vars[2 if fw != 'torch' else 0], vars[3 if fw != 'torch' else 1], framework=fw), 1)
            q_target_tp1 = np.max(fc(fc(input_[SampleBatch.NEXT_OBS], vars_t[0 if fw != 'torch' else 2], vars_t[1 if fw != 'torch' else 3], framework=fw), vars_t[2 if fw != 'torch' else 0], vars_t[3 if fw != 'torch' else 1], framework=fw), 1)
            td_error = q_t - config.gamma * input_[SampleBatch.REWARDS] + q_target_tp1
            expected_loss = huber_loss(td_error).mean()
            if fw == 'torch':
                input_ = policy._lazy_tensor_dict(input_)
            if fw == 'tf':
                out = policy.get_session().run(policy._loss, feed_dict=policy._get_loss_inputs_dict(input_, shuffle=False))
            else:
                out = (SimpleQTorchPolicy if fw == 'torch' else SimpleQTF2Policy).loss(policy, policy.model, None, input_)
            check(out, expected_loss, decimals=1)

    def test_simple_q_lr_schedule(self):
        if False:
            for i in range(10):
                print('nop')
        'Test PG with learning rate schedule.'
        config = simple_q.SimpleQConfig()
        config.reporting(min_sample_timesteps_per_iteration=10, min_train_timesteps_per_iteration=10, min_time_s_per_iteration=0)
        config.rollouts(num_rollout_workers=1, rollout_fragment_length=50)
        config.training(lr=0.2, lr_schedule=[[0, 0.2], [500, 0.001]])

        def _step_n_times(algo, n: int):
            if False:
                return 10
            'Step trainer n times.\n\n            Returns:\n                learning rate at the end of the execution.\n            '
            for _ in range(n):
                results = algo.train()
            return results['info'][LEARNER_INFO][DEFAULT_POLICY_ID][LEARNER_STATS_KEY]['cur_lr']
        for _ in framework_iterator(config):
            algo = config.build(env='CartPole-v1')
            lr = _step_n_times(algo, 1)
            self.assertGreaterEqual(lr, 0.15)
            lr = _step_n_times(algo, 8)
            self.assertLessEqual(float(lr), 0.5)
            lr = _step_n_times(algo, 2)
            self.assertAlmostEqual(lr, 0.001)
            algo.stop()
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))