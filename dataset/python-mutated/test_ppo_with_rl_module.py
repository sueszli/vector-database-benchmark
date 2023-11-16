import unittest
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo.ppo_learner import LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo.tests.test_ppo import PENDULUM_FAKE_BATCH
from ray.rllib.core.learner.learner import LEARNER_RESULTS_CURR_LR_KEY
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.test_utils import check, check_compute_single_action, check_train_results, framework_iterator

def get_model_config(framework, lstm=False):
    if False:
        for i in range(10):
            print('nop')
    return dict(use_lstm=True, lstm_use_prev_action=True, lstm_use_prev_reward=True, lstm_cell_size=10, max_seq_len=20) if lstm else {'use_lstm': False}

class MyCallbacks(DefaultCallbacks):

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if False:
            while True:
                i = 10
        stats = result['info'][LEARNER_INFO][DEFAULT_POLICY_ID]
        check(stats[LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY], 0.05 if algorithm.iteration == 1 else 0.0)
        check(stats[LEARNER_RESULTS_CURR_LR_KEY], 7.5e-06 if algorithm.iteration == 1 else 5e-06)
        optim = algorithm.learner_group._learner.get_optimizer()
        actual_optimizer_lr = optim.param_groups[0]['lr'] if algorithm.config.framework_str == 'torch' else optim.lr
        check(stats[LEARNER_RESULTS_CURR_LR_KEY], actual_optimizer_lr)

class TestPPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        ray.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        ray.shutdown()

    def test_ppo_compilation_and_schedule_mixins(self):
        if False:
            i = 10
            return i + 15
        'Test whether PPO can be built with all frameworks.'
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=True).training(num_sgd_iter=2, lr=[[0, 1e-05], [512, 0.0]], entropy_coeff=[[0, 0.1], [256, 0.0]], train_batch_size=128).rollouts(num_rollout_workers=1, enable_connectors=True).callbacks(MyCallbacks)
        num_iterations = 2
        for fw in framework_iterator(config, frameworks=('torch', 'tf2')):
            for env in ['CartPole-v1', 'Pendulum-v1', 'ALE/Breakout-v5']:
                print('Env={}'.format(env))
                for lstm in [False, True]:
                    print('LSTM={}'.format(lstm))
                    config.training(model=get_model_config(fw, lstm=lstm))
                    algo = config.build(env=env)
                    learner = algo.learner_group._learner
                    optim = learner.get_optimizer()
                    lr = optim.param_groups[0]['lr'] if fw == 'torch' else optim.lr
                    check(lr, config.lr[0][1])
                    entropy_coeff = learner.entropy_coeff_schedulers_per_module[DEFAULT_POLICY_ID].get_current_value()
                    check(entropy_coeff, 0.1)
                    for i in range(num_iterations):
                        results = algo.train()
                        check_train_results(results)
                        print(results)
                    check_compute_single_action(algo, include_prev_action_reward=True, include_state=lstm)
                    algo.stop()

    def test_ppo_exploration_setup(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests, whether PPO runs with different exploration setups.'
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=True).environment('FrozenLake-v1', env_config={'is_slippery': False, 'map_name': '4x4'}).rollouts(num_rollout_workers=1, enable_connectors=True)
        obs = np.array(0)
        for _ in framework_iterator(config, frameworks=('torch', 'tf2')):
            algo = config.build()
            a_ = algo.compute_single_action(obs, explore=False, prev_action=np.array(2), prev_reward=np.array(1.0))
            for _ in range(50):
                a = algo.compute_single_action(obs, explore=False, prev_action=np.array(2), prev_reward=np.array(1.0))
                check(a, a_)
            actions = []
            for _ in range(300):
                actions.append(algo.compute_single_action(obs, prev_action=np.array(2), prev_reward=np.array(1.0)))
            check(np.mean(actions), 1.5, atol=0.2)
            algo.stop()

    def test_ppo_free_log_std_with_rl_modules(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the free log std option works.'
        config = ppo.PPOConfig().experimental(_enable_new_api_stack=True).environment('Pendulum-v1').rollouts(num_rollout_workers=1).training(gamma=0.99, model=dict(fcnet_hiddens=[10], fcnet_activation='linear', free_log_std=True, vf_share_layers=True))
        for fw in framework_iterator(config, frameworks=('torch', 'tf2')):
            algo = config.build()
            policy = algo.get_policy()
            learner = algo.learner_group._learner
            module = learner.module[DEFAULT_POLICY_ID]
            if fw == 'torch':
                matching = [v for (n, v) in module.named_parameters() if 'log_std' in n]
            else:
                matching = [v for v in module.trainable_variables if 'log_std' in str(v)]
            assert len(matching) == 1, matching
            log_std_var = matching[0]

            def get_value():
                if False:
                    print('Hello World!')
                if fw == 'torch':
                    return log_std_var.detach().cpu().numpy()[0]
                else:
                    return log_std_var.numpy()[0]
            init_std = get_value()
            assert init_std == 0.0, init_std
            batch = compute_gae_for_sample_batch(policy, PENDULUM_FAKE_BATCH.copy())
            batch = policy._lazy_tensor_dict(batch)
            algo.learner_group.update(batch.as_multi_agent())
            post_std = get_value()
            assert post_std != 0.0, post_std
            algo.stop()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))