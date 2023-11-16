import unittest
import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF2Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, Postprocessing
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.rllib.utils.numpy import fc
from ray.rllib.utils.test_utils import check, check_compute_single_action, check_off_policyness, check_train_results, framework_iterator, check_inference_w_connectors
CARTPOLE_FAKE_BATCH = SampleBatch({SampleBatch.OBS: np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=np.float32), SampleBatch.ACTIONS: np.array([0, 1, 1]), SampleBatch.PREV_ACTIONS: np.array([0, 1, 1]), SampleBatch.REWARDS: np.array([1.0, -1.0, 0.5], dtype=np.float32), SampleBatch.PREV_REWARDS: np.array([1.0, -1.0, 0.5], dtype=np.float32), SampleBatch.TERMINATEDS: np.array([False, False, True]), SampleBatch.TRUNCATEDS: np.array([False, False, False]), SampleBatch.VF_PREDS: np.array([0.5, 0.6, 0.7], dtype=np.float32), SampleBatch.ACTION_DIST_INPUTS: np.array([[-2.0, 0.5], [-3.0, -0.3], [-0.1, 2.5]], dtype=np.float32), SampleBatch.ACTION_LOGP: np.array([-0.5, -0.1, -0.2], dtype=np.float32), SampleBatch.EPS_ID: np.array([0, 0, 0]), SampleBatch.AGENT_INDEX: np.array([0, 0, 0])})
PENDULUM_FAKE_BATCH = SampleBatch({SampleBatch.OBS: np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32), SampleBatch.ACTIONS: np.array([0.1, 0.2, 0.3], dtype=np.float32), SampleBatch.PREV_ACTIONS: np.array([0.3, 0.4], dtype=np.float32), SampleBatch.REWARDS: np.array([1.0, -1.0, 0.5], dtype=np.float32), SampleBatch.PREV_REWARDS: np.array([1.0, -1.0, 0.5], dtype=np.float32), SampleBatch.TERMINATEDS: np.array([False, False, True]), SampleBatch.TRUNCATEDS: np.array([False, False, False]), SampleBatch.VF_PREDS: np.array([0.5, 0.6, 0.7], dtype=np.float32), SampleBatch.ACTION_DIST_INPUTS: np.array([[0.1, 0.0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]], dtype=np.float32), SampleBatch.ACTION_LOGP: np.array([-0.5, -0.1, -0.2], dtype=np.float32), SampleBatch.EPS_ID: np.array([0, 0, 0]), SampleBatch.AGENT_INDEX: np.array([0, 0, 0])})

class MyCallbacks(DefaultCallbacks):

    @staticmethod
    def _check_lr_torch(policy, policy_id):
        if False:
            while True:
                i = 10
        for (j, opt) in enumerate(policy._optimizers):
            for p in opt.param_groups:
                assert p['lr'] == policy.cur_lr, 'LR scheduling error!'

    @staticmethod
    def _check_lr_tf(policy, policy_id):
        if False:
            i = 10
            return i + 15
        lr = policy.cur_lr
        sess = policy.get_session()
        if sess:
            lr = sess.run(lr)
            optim_lr = sess.run(policy._optimizer._lr)
        else:
            lr = lr.numpy()
            optim_lr = policy._optimizer.lr.numpy()
        assert lr == optim_lr, 'LR scheduling error!'

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        stats = result['info'][LEARNER_INFO][DEFAULT_POLICY_ID][LEARNER_STATS_KEY]
        check(stats['cur_lr'], 5e-05 if algorithm.iteration == 1 else 0.0)
        check(stats['entropy_coeff'], 0.1 if algorithm.iteration == 1 else 0.05)
        algorithm.workers.foreach_policy(self._check_lr_torch if algorithm.config['framework'] == 'torch' else self._check_lr_tf)

class TestPPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        ray.shutdown()

    def test_ppo_compilation_w_connectors(self):
        if False:
            i = 10
            return i + 15
        'Test whether PPO can be built with all frameworks w/ connectors.'
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=False).training(num_sgd_iter=2, lr_schedule=[[0, 5e-05], [128, 0.0]], entropy_coeff=100.0, entropy_coeff_schedule=[[0, 0.1], [256, 0.0]], train_batch_size=128, model=dict(lstm_cell_size=10, max_seq_len=20)).rollouts(num_rollout_workers=1, compress_observations=True, enable_connectors=True).callbacks(MyCallbacks).evaluation(evaluation_duration=2, evaluation_duration_unit='episodes', evaluation_num_workers=1)
        num_iterations = 2
        for fw in framework_iterator(config):
            for env in ['FrozenLake-v1', 'ALE/MsPacman-v5']:
                print('Env={}'.format(env))
                for lstm in [False, True]:
                    print('LSTM={}'.format(lstm))
                    config.training(model=dict(use_lstm=lstm, lstm_use_prev_action=lstm, lstm_use_prev_reward=lstm))
                    algo = config.build(env=env)
                    policy = algo.get_policy()
                    entropy_coeff = algo.get_policy().entropy_coeff
                    lr = policy.cur_lr
                    if fw == 'tf':
                        (entropy_coeff, lr) = policy.get_session().run([entropy_coeff, lr])
                    check(entropy_coeff, 0.1)
                    check(lr, config.lr)
                    for i in range(num_iterations):
                        results = algo.train()
                        check_train_results(results)
                        print(results)
                    algo.evaluate()
                    check_inference_w_connectors(policy, env_name=env)
                    algo.stop()

    def test_ppo_compilation_and_schedule_mixins(self):
        if False:
            i = 10
            return i + 15
        'Test whether PPO can be built with all frameworks.'
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=False).training(lr_schedule=[[0, 5e-05], [256, 0.0]], entropy_coeff=100.0, entropy_coeff_schedule=[[0, 0.1], [512, 0.0]], train_batch_size=256, sgd_minibatch_size=128, num_sgd_iter=2, model=dict(lstm_cell_size=10, max_seq_len=20)).rollouts(num_rollout_workers=1, compress_observations=True).callbacks(MyCallbacks)
        num_iterations = 2
        for fw in framework_iterator(config):
            for env in ['FrozenLake-v1', 'ALE/MsPacman-v5']:
                print('Env={}'.format(env))
                for lstm in [False, True]:
                    print('LSTM={}'.format(lstm))
                    config.training(model=dict(use_lstm=lstm, lstm_use_prev_action=lstm, lstm_use_prev_reward=lstm))
                    algo = config.build(env=env)
                    policy = algo.get_policy()
                    entropy_coeff = algo.get_policy().entropy_coeff
                    lr = policy.cur_lr
                    if fw == 'tf':
                        (entropy_coeff, lr) = policy.get_session().run([entropy_coeff, lr])
                    check(entropy_coeff, 0.1)
                    check(lr, config.lr)
                    for i in range(num_iterations):
                        results = algo.train()
                        print(results)
                        check_train_results(results)
                        off_policy_ness = check_off_policyness(results, lower_limit=1.5, upper_limit=1.5)
                        print(f"off-policy'ness={off_policy_ness}")
                    check_compute_single_action(algo, include_prev_action_reward=True, include_state=lstm)
                    algo.stop()

    def test_ppo_exploration_setup(self):
        if False:
            while True:
                i = 10
        'Tests, whether PPO runs with different exploration setups.'
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=True).environment('FrozenLake-v1', env_config={'is_slippery': False, 'map_name': '4x4'}).rollouts(num_rollout_workers=0)
        obs = np.array(0)
        for fw in framework_iterator(config):
            algo = config.build()
            a_ = algo.compute_single_action(obs, explore=False, prev_action=np.array(2), prev_reward=np.array(1.0))
            config.validate()
            if not config._enable_new_api_stack and fw != 'tf':
                last_out = algo.get_policy().model.last_output()
                if fw == 'torch':
                    check(a_, np.argmax(last_out.detach().cpu().numpy(), 1)[0])
                else:
                    check(a_, np.argmax(last_out.numpy(), 1)[0])
            for _ in range(50):
                a = algo.compute_single_action(obs, explore=False, prev_action=np.array(2), prev_reward=np.array(1.0))
                check(a, a_)
            actions = []
            for _ in range(300):
                actions.append(algo.compute_single_action(obs, prev_action=np.array(2), prev_reward=np.array(1.0)))
            check(np.mean(actions), 1.5, atol=0.2)
            algo.stop()

    def test_ppo_free_log_std(self):
        if False:
            return 10
        'Tests the free log std option works.\n\n        This test is overfitted to the old ModelV2 stack (e.g.\n        policy.model.trainable_variables is not callable in the new stack)\n        # TODO (Kourosh) we should create a new test for the new RLModule stack.\n        '
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=False).environment('CartPole-v1').rollouts(num_rollout_workers=0).training(gamma=0.99, model=dict(fcnet_hiddens=[10], fcnet_activation='linear', free_log_std=True, vf_share_layers=True))
        for (fw, sess) in framework_iterator(config, session=True):
            algo = config.build()
            policy = algo.get_policy()
            if fw == 'torch':
                matching = [v for (n, v) in policy.model.named_parameters() if 'log_std' in n]
            else:
                matching = [v for v in policy.model.trainable_variables() if 'log_std' in str(v)]
            assert len(matching) == 1, matching
            log_std_var = matching[0]

            def get_value(fw=fw, policy=policy, log_std_var=log_std_var):
                if False:
                    for i in range(10):
                        print('nop')
                if fw == 'tf':
                    return policy.get_session().run(log_std_var)[0]
                elif fw == 'torch':
                    return log_std_var.detach().cpu().numpy()[0]
                else:
                    return log_std_var.numpy()[0]
            init_std = get_value()
            assert init_std == 0.0, init_std
            batch = compute_gae_for_sample_batch(policy, CARTPOLE_FAKE_BATCH.copy())
            if fw == 'torch':
                batch = policy._lazy_tensor_dict(batch)
            policy.learn_on_batch(batch)
            post_std = get_value()
            assert post_std != 0.0, post_std
            algo.stop()

    def test_ppo_loss_function(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the PPO loss function math.\n\n        This test is overfitted to the old ModelV2 stack (e.g.\n        policy.model.trainable_variables is not callable in the new stack)\n        # TODO (Kourosh) we should create a new test for the new RLModule stack.\n        '
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=False).environment('CartPole-v1').rollouts(num_rollout_workers=0).training(gamma=0.99, model=dict(fcnet_hiddens=[10], fcnet_activation='linear', vf_share_layers=True))
        for (fw, sess) in framework_iterator(config, session=True):
            algo = config.build()
            policy = algo.get_policy()
            if fw == 'torch':
                matching = [v for (n, v) in policy.model.named_parameters() if 'log_std' in n]
            else:
                matching = [v for v in policy.model.trainable_variables() if 'log_std' in str(v)]
            assert len(matching) == 0, matching
            train_batch = compute_gae_for_sample_batch(policy, CARTPOLE_FAKE_BATCH.copy())
            if fw == 'torch':
                train_batch = policy._lazy_tensor_dict(train_batch)
            check(train_batch[Postprocessing.VALUE_TARGETS], [0.50005, -0.505, 0.5])
            if fw == 'tf2':
                PPOTF2Policy.loss(policy, policy.model, Categorical, train_batch)
            elif fw == 'torch':
                PPOTorchPolicy.loss(policy, policy.model, policy.dist_class, train_batch)
            vars = policy.model.variables() if fw != 'torch' else list(policy.model.parameters())
            if fw == 'tf':
                vars = policy.get_session().run(vars)
            expected_shared_out = fc(train_batch[SampleBatch.CUR_OBS], vars[0 if fw != 'torch' else 2], vars[1 if fw != 'torch' else 3], framework=fw)
            expected_logits = fc(expected_shared_out, vars[2 if fw != 'torch' else 0], vars[3 if fw != 'torch' else 1], framework=fw)
            expected_value_outs = fc(expected_shared_out, vars[4], vars[5], framework=fw)
            (kl, entropy, pg_loss, vf_loss, overall_loss) = self._ppo_loss_helper(policy, policy.model, Categorical if fw != 'torch' else TorchCategorical, train_batch, expected_logits, expected_value_outs, sess=sess)
            if sess:
                policy_sess = policy.get_session()
                (k, e, pl, v, tl) = policy_sess.run([policy._mean_kl_loss, policy._mean_entropy, policy._mean_policy_loss, policy._mean_vf_loss, policy._total_loss], feed_dict=policy._get_loss_inputs_dict(train_batch, shuffle=False))
                check(k, kl)
                check(e, entropy)
                check(pl, np.mean(-pg_loss))
                check(v, np.mean(vf_loss), decimals=4)
                check(tl, overall_loss, decimals=4)
            elif fw == 'torch':
                check(policy.model.tower_stats['mean_kl_loss'], kl)
                check(policy.model.tower_stats['mean_entropy'], entropy)
                check(policy.model.tower_stats['mean_policy_loss'], np.mean(-pg_loss))
                check(policy.model.tower_stats['mean_vf_loss'], np.mean(vf_loss), decimals=4)
                check(policy.model.tower_stats['total_loss'], overall_loss, decimals=4)
            else:
                check(policy._mean_kl_loss, kl)
                check(policy._mean_entropy, entropy)
                check(policy._mean_policy_loss, np.mean(-pg_loss))
                check(policy._mean_vf_loss, np.mean(vf_loss), decimals=4)
                check(policy._total_loss, overall_loss, decimals=4)
            algo.stop()

    def _ppo_loss_helper(self, policy, model, dist_class, train_batch, logits, vf_outs, sess=None):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the expected PPO loss (components) given Policy,\n        Model, distribution, some batch, logits & vf outputs, using numpy.\n        '
        dist = dist_class(logits, policy.model)
        dist_prev = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], policy.model)
        expected_logp = dist.logp(train_batch[SampleBatch.ACTIONS])
        if isinstance(model, TorchModelV2):
            train_batch.set_get_interceptor(None)
            expected_rho = np.exp(expected_logp.detach().cpu().numpy() - train_batch[SampleBatch.ACTION_LOGP])
            kl = np.mean(dist_prev.kl(dist).detach().cpu().numpy())
            entropy = np.mean(dist.entropy().detach().cpu().numpy())
        else:
            if sess:
                expected_logp = sess.run(expected_logp)
            expected_rho = np.exp(expected_logp - train_batch[SampleBatch.ACTION_LOGP])
            kl = dist_prev.kl(dist)
            if sess:
                kl = sess.run(kl)
            kl = np.mean(kl)
            entropy = dist.entropy()
            if sess:
                entropy = sess.run(entropy)
            entropy = np.mean(entropy)
        pg_loss = np.minimum(train_batch[Postprocessing.ADVANTAGES] * expected_rho, train_batch[Postprocessing.ADVANTAGES] * np.clip(expected_rho, 1 - policy.config['clip_param'], 1 + policy.config['clip_param']))
        vf_loss1 = np.power(vf_outs - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = train_batch[SampleBatch.VF_PREDS] + np.clip(vf_outs - train_batch[SampleBatch.VF_PREDS], -policy.config['vf_clip_param'], policy.config['vf_clip_param'])
        vf_loss2 = np.power(vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = np.maximum(vf_loss1, vf_loss2)
        if sess:
            policy_sess = policy.get_session()
            (kl_coeff, entropy_coeff) = policy_sess.run([policy.kl_coeff, policy.entropy_coeff])
        else:
            (kl_coeff, entropy_coeff) = (policy.kl_coeff, policy.entropy_coeff)
        overall_loss = np.mean(-pg_loss + kl_coeff * kl + policy.config['vf_loss_coeff'] * vf_loss - entropy_coeff * entropy)
        return (kl, entropy, pg_loss, vf_loss, overall_loss)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))