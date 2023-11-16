import copy
import re
import unittest
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from rllib_ddpg.ddpg import DDPGConfig
import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy, fc, huber_loss, l2_loss, relu, sigmoid
from ray.rllib.utils.replay_buffers.utils import patch_buffer_with_fake_sampling_method
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.test_utils import check, check_compute_single_action, check_train_results, framework_iterator
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
(tf1, tf, tfv) = try_import_tf()
(torch, _) = try_import_torch()

class SimpleEnv(gym.Env):

    def __init__(self, config):
        if False:
            while True:
                i = 10
        self._skip_env_checking = True
        if config.get('simplex_actions', False):
            self.action_space = Simplex((2,))
        else:
            self.action_space = Box(0.0, 1.0, (1,))
        self.observation_space = Box(0.0, 1.0, (1,))
        self.max_steps = config.get('max_steps', 100)
        self.state = None
        self.steps = None

    def reset(self, *, seed=None, options=None):
        if False:
            print('Hello World!')
        self.state = self.observation_space.sample()
        self.steps = 0
        return (self.state, {})

    def step(self, action):
        if False:
            print('Hello World!')
        self.steps += 1
        [rew] = 1.0 - np.abs(np.max(action) - self.state)
        terminated = False
        truncated = self.steps >= self.max_steps
        self.state = self.observation_space.sample()
        return (self.state, rew, terminated, truncated, {})

class TestDDPG(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            while True:
                i = 10
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        ray.shutdown()

    def test_ddpg_compilation(self):
        if False:
            i = 10
            return i + 15
        'Test whether DDPG can be built with both frameworks.'
        config = DDPGConfig().training(num_steps_sampled_before_learning_starts=0).rollouts(num_rollout_workers=0, num_envs_per_worker=2).exploration(exploration_config={'random_timesteps': 100})
        num_iterations = 1
        for _ in framework_iterator(config, with_eager_tracing=True):
            algo = config.build(env='Pendulum-v1')
            for i in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            check_compute_single_action(algo)
            pol = algo.get_policy()
            if config.framework_str == 'tf':
                a = pol.get_session().run(pol.global_step)
            else:
                a = pol.global_step
            check(convert_to_numpy(a), 500)
            algo.stop()

    def test_ddpg_exploration_and_with_random_prerun(self):
        if False:
            return 10
        "Tests DDPG's Exploration (w/ random actions for n timesteps)."
        core_config = DDPGConfig().environment('Pendulum-v1').rollouts(num_rollout_workers=0).training(num_steps_sampled_before_learning_starts=0)
        obs = np.array([0.0, 0.1, -0.1])
        for _ in framework_iterator(core_config):
            config = copy.deepcopy(core_config)
            algo = config.build()
            a_ = algo.compute_single_action(obs, explore=False)
            check(convert_to_numpy(algo.get_policy().global_timestep), 1)
            for i in range(50):
                a = algo.compute_single_action(obs, explore=False)
                check(convert_to_numpy(algo.get_policy().global_timestep), i + 2)
                check(a, a_)
            actions = []
            for i in range(50):
                actions.append(algo.compute_single_action(obs))
                check(convert_to_numpy(algo.get_policy().global_timestep), i + 52)
            check(np.std(actions), 0.0, false=True)
            algo.stop()
            config.exploration(exploration_config={'random_timesteps': 50, 'ou_base_scale': 0.001, 'initial_scale': 0.001, 'final_scale': 0.001})
            algo = config.build()
            deterministic_action = algo.compute_single_action(obs, explore=False)
            check(convert_to_numpy(algo.get_policy().global_timestep), 1)
            random_a = []
            for i in range(1, 50):
                random_a.append(algo.compute_single_action(obs, explore=True))
                check(convert_to_numpy(algo.get_policy().global_timestep), i + 1)
                check(random_a[-1], deterministic_action, false=True)
            self.assertTrue(np.std(random_a) > 0.5)
            for i in range(50):
                a = algo.compute_single_action(obs, explore=True)
                check(convert_to_numpy(algo.get_policy().global_timestep), i + 51)
                check(a, deterministic_action, rtol=0.1)
            for i in range(50):
                a = algo.compute_single_action(obs, explore=False)
                check(convert_to_numpy(algo.get_policy().global_timestep), i + 101)
                check(a, deterministic_action)
            algo.stop()

    def test_ddpg_loss_function(self):
        if False:
            i = 10
            return i + 15
        'Tests DDPG loss function results across all frameworks.'
        config = DDPGConfig().training(num_steps_sampled_before_learning_starts=0)
        config.seed = 42
        config.num_rollout_workers = 0
        config.twin_q = True
        config.use_huber = True
        config.huber_threshold = 1.0
        config.gamma = 0.99
        config.l2_reg = 1e-10
        config.replay_buffer_config = {'type': 'MultiAgentReplayBuffer', 'capacity': 50000}
        config.num_steps_sampled_before_learning_starts = 0
        config.actor_hiddens = [10]
        config.critic_hiddens = [10]
        config.min_time_s_per_iteration = 0
        config.min_sample_timesteps_per_iteration = 100
        map_ = {'default_policy/actor_hidden_0/kernel': 'policy_model.action_0._model.0.weight', 'default_policy/actor_hidden_0/bias': 'policy_model.action_0._model.0.bias', 'default_policy/actor_out/kernel': 'policy_model.action_out._model.0.weight', 'default_policy/actor_out/bias': 'policy_model.action_out._model.0.bias', 'default_policy/sequential/q_hidden_0/kernel': 'q_model.q_hidden_0._model.0.weight', 'default_policy/sequential/q_hidden_0/bias': 'q_model.q_hidden_0._model.0.bias', 'default_policy/sequential/q_out/kernel': 'q_model.q_out._model.0.weight', 'default_policy/sequential/q_out/bias': 'q_model.q_out._model.0.bias', 'default_policy/sequential_1/twin_q_hidden_0/kernel': 'twin_q_model.twin_q_hidden_0._model.0.weight', 'default_policy/sequential_1/twin_q_hidden_0/bias': 'twin_q_model.twin_q_hidden_0._model.0.bias', 'default_policy/sequential_1/twin_q_out/kernel': 'twin_q_model.twin_q_out._model.0.weight', 'default_policy/sequential_1/twin_q_out/bias': 'twin_q_model.twin_q_out._model.0.bias', 'default_policy/actor_hidden_0_1/kernel': 'policy_model.action_0._model.0.weight', 'default_policy/actor_hidden_0_1/bias': 'policy_model.action_0._model.0.bias', 'default_policy/actor_out_1/kernel': 'policy_model.action_out._model.0.weight', 'default_policy/actor_out_1/bias': 'policy_model.action_out._model.0.bias', 'default_policy/sequential_2/q_hidden_0/kernel': 'q_model.q_hidden_0._model.0.weight', 'default_policy/sequential_2/q_hidden_0/bias': 'q_model.q_hidden_0._model.0.bias', 'default_policy/sequential_2/q_out/kernel': 'q_model.q_out._model.0.weight', 'default_policy/sequential_2/q_out/bias': 'q_model.q_out._model.0.bias', 'default_policy/sequential_3/twin_q_hidden_0/kernel': 'twin_q_model.twin_q_hidden_0._model.0.weight', 'default_policy/sequential_3/twin_q_hidden_0/bias': 'twin_q_model.twin_q_hidden_0._model.0.bias', 'default_policy/sequential_3/twin_q_out/kernel': 'twin_q_model.twin_q_out._model.0.weight', 'default_policy/sequential_3/twin_q_out/bias': 'twin_q_model.twin_q_out._model.0.bias'}
        env = SimpleEnv
        batch_size = 100
        obs_size = (batch_size, 1)
        actions = np.random.random(size=(batch_size, 1))
        input_ = self._get_batch_helper(obs_size, actions, batch_size)
        prev_fw_loss = weights_dict = None
        (expect_c, expect_a, expect_t) = (None, None, None)
        tf_updated_weights = []
        tf_inputs = []
        for (fw, sess) in framework_iterator(config, frameworks=('tf', 'torch'), session=True):
            algo = config.build(env=env)
            policy = algo.get_policy()
            p_sess = None
            if sess:
                p_sess = policy.get_session()
            if weights_dict is None:
                assert fw == 'tf'
                weights_dict_list = policy.model.variables() + policy.target_model.variables()
                with p_sess.graph.as_default():
                    collector = ray.experimental.tf_utils.TensorFlowVariables([], p_sess, weights_dict_list)
                    weights_dict = collector.get_weights()
            else:
                assert fw == 'torch'
                model_dict = self._translate_weights_to_torch(weights_dict, map_)
                policy.model.load_state_dict(model_dict)
                policy.target_model.load_state_dict(model_dict)
            if fw == 'torch':
                input_ = policy._lazy_tensor_dict(input_)
                input_ = {k: input_[k] for k in input_.keys()}
            if expect_c is None:
                (expect_c, expect_a, expect_t) = self._ddpg_loss_helper(input_, weights_dict, sorted(weights_dict.keys()), fw, gamma=config.gamma, huber_threshold=config.huber_threshold, l2_reg=config.l2_reg, sess=sess)
            if fw == 'tf':
                (c, a, t, tf_c_grads, tf_a_grads) = p_sess.run([policy.critic_loss, policy.actor_loss, policy.td_error, policy._critic_optimizer.compute_gradients(policy.critic_loss, policy.model.q_variables()), policy._actor_optimizer.compute_gradients(policy.actor_loss, policy.model.policy_variables())], feed_dict=policy._get_loss_inputs_dict(input_, shuffle=False))
                check(c, expect_c)
                check(a, expect_a)
                check(t, expect_t)
                tf_c_grads = [g for (g, v) in tf_c_grads]
                tf_a_grads = [g for (g, v) in tf_a_grads]
            elif fw == 'torch':
                policy.loss(policy.model, None, input_)
                (c, a, t) = (policy.get_tower_stats('critic_loss')[0], policy.get_tower_stats('actor_loss')[0], policy.get_tower_stats('td_error')[0])
                check(c, expect_c)
                check(a, expect_a)
                check(t, expect_t)
                policy._actor_optimizer.zero_grad()
                assert all((v.grad is None for v in policy.model.q_variables()))
                assert all((v.grad is None for v in policy.model.policy_variables()))
                a.backward()
                assert not any((v.grad is None for v in policy.model.q_variables()[:4]))
                assert all((v.grad is None for v in policy.model.q_variables()[4:]))
                assert not all((torch.mean(v.grad) == 0 for v in policy.model.policy_variables()))
                assert not all((torch.min(v.grad) == 0 for v in policy.model.policy_variables()))
                torch_a_grads = [v.grad for v in policy.model.policy_variables()]
                for (tf_g, torch_g) in zip(tf_a_grads, torch_a_grads):
                    if tf_g.shape != torch_g.shape:
                        check(tf_g, np.transpose(torch_g.cpu()))
                    else:
                        check(tf_g, torch_g)
                policy._critic_optimizer.zero_grad()
                assert all((v.grad is None or torch.mean(v.grad) == 0.0 for v in policy.model.q_variables()))
                assert all((v.grad is None or torch.min(v.grad) == 0.0 for v in policy.model.q_variables()))
                c.backward()
                assert not all((torch.mean(v.grad) == 0 for v in policy.model.q_variables()))
                assert not all((torch.min(v.grad) == 0 for v in policy.model.q_variables()))
                torch_c_grads = [v.grad for v in policy.model.q_variables()]
                for (tf_g, torch_g) in zip(tf_c_grads, torch_c_grads):
                    if tf_g.shape != torch_g.shape:
                        check(tf_g, np.transpose(torch_g.cpu()))
                    else:
                        check(tf_g, torch_g)
                torch_a_grads = [v.grad for v in policy.model.policy_variables()]
                for (tf_g, torch_g) in zip(tf_a_grads, torch_a_grads):
                    if tf_g.shape != torch_g.shape:
                        check(tf_g, np.transpose(torch_g.cpu()))
                    else:
                        check(tf_g, torch_g)
            if prev_fw_loss is not None:
                check(c, prev_fw_loss[0])
                check(a, prev_fw_loss[1])
                check(t, prev_fw_loss[2])
            prev_fw_loss = (c, a, t)
            for update_iteration in range(6):
                print('train iteration {}'.format(update_iteration))
                if fw == 'tf':
                    in_ = self._get_batch_helper(obs_size, actions, batch_size)
                    tf_inputs.append(in_)
                    buf = algo.local_replay_buffer
                    patch_buffer_with_fake_sampling_method(buf, in_)
                    algo.train()
                    updated_weights = policy.get_weights()
                    if tf_updated_weights:
                        check(updated_weights['default_policy/actor_hidden_0/kernel'], tf_updated_weights[-1]['default_policy/actor_hidden_0/kernel'], false=True)
                    tf_updated_weights.append(updated_weights)
                else:
                    tf_weights = tf_updated_weights[update_iteration]
                    in_ = tf_inputs[update_iteration]
                    buf = algo.local_replay_buffer
                    patch_buffer_with_fake_sampling_method(buf, in_)
                    algo.train()
                    for tf_key in tf_weights.keys():
                        tf_var = tf_weights[tf_key]
                        if re.search('actor_out_1|actor_hidden_0_1|sequential_[23]', tf_key):
                            torch_var = policy.target_model.state_dict()[map_[tf_key]]
                        else:
                            torch_var = policy.model.state_dict()[map_[tf_key]]
                        if tf_var.shape != torch_var.shape:
                            check(tf_var, np.transpose(torch_var.cpu()), atol=0.1)
                        else:
                            check(tf_var, torch_var, atol=0.1)
            algo.stop()

    def _get_batch_helper(self, obs_size, actions, batch_size):
        if False:
            for i in range(10):
                print('nop')
        return SampleBatch({SampleBatch.CUR_OBS: np.random.random(size=obs_size), SampleBatch.ACTIONS: actions, SampleBatch.REWARDS: np.random.random(size=(batch_size,)), SampleBatch.TERMINATEDS: np.random.choice([True, False], size=(batch_size,)), SampleBatch.NEXT_OBS: np.random.random(size=obs_size), 'weights': np.ones(shape=(batch_size,))})

    def _ddpg_loss_helper(self, train_batch, weights, ks, fw, gamma, huber_threshold, l2_reg, sess):
        if False:
            while True:
                i = 10
        'Emulates DDPG loss functions for tf and torch.'
        model_out_t = train_batch[SampleBatch.CUR_OBS]
        target_model_out_tp1 = train_batch[SampleBatch.NEXT_OBS]
        policy_t = sigmoid(2.0 * fc(relu(fc(model_out_t, weights[ks[1]], weights[ks[0]], framework=fw)), weights[ks[5]], weights[ks[4]], framework=fw))
        policy_tp1 = sigmoid(2.0 * fc(relu(fc(target_model_out_tp1, weights[ks[3]], weights[ks[2]], framework=fw)), weights[ks[7]], weights[ks[6]], framework=fw))
        policy_tp1_smoothed = policy_tp1
        q_t = fc(relu(fc(np.concatenate([model_out_t, train_batch[SampleBatch.ACTIONS]], -1), weights[ks[9]], weights[ks[8]], framework=fw)), weights[ks[11]], weights[ks[10]], framework=fw)
        twin_q_t = fc(relu(fc(np.concatenate([model_out_t, train_batch[SampleBatch.ACTIONS]], -1), weights[ks[13]], weights[ks[12]], framework=fw)), weights[ks[15]], weights[ks[14]], framework=fw)
        q_t_det_policy = fc(relu(fc(np.concatenate([model_out_t, policy_t], -1), weights[ks[9]], weights[ks[8]], framework=fw)), weights[ks[11]], weights[ks[10]], framework=fw)
        q_tp1 = fc(relu(fc(np.concatenate([target_model_out_tp1, policy_tp1_smoothed], -1), weights[ks[17]], weights[ks[16]], framework=fw)), weights[ks[19]], weights[ks[18]], framework=fw)
        twin_q_tp1 = fc(relu(fc(np.concatenate([target_model_out_tp1, policy_tp1_smoothed], -1), weights[ks[21]], weights[ks[20]], framework=fw)), weights[ks[23]], weights[ks[22]], framework=fw)
        q_t_selected = np.squeeze(q_t, axis=-1)
        twin_q_t_selected = np.squeeze(twin_q_t, axis=-1)
        q_tp1 = np.minimum(q_tp1, twin_q_tp1)
        q_tp1_best = np.squeeze(q_tp1, axis=-1)
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        if fw == 'torch':
            dones = dones.float().numpy()
            rewards = rewards.numpy()
        q_tp1_best_masked = (1.0 - dones) * q_tp1_best
        q_t_selected_target = rewards + gamma * q_tp1_best_masked
        td_error = q_t_selected - q_t_selected_target
        twin_td_error = twin_q_t_selected - q_t_selected_target
        errors = huber_loss(td_error, huber_threshold) + huber_loss(twin_td_error, huber_threshold)
        critic_loss = np.mean(errors)
        actor_loss = -np.mean(q_t_det_policy)
        for (name, var) in weights.items():
            if re.match('default_policy/actor_(hidden_0|out)/kernel', name):
                actor_loss += l2_reg * l2_loss(var)
            elif re.match('default_policy/sequential(_1)?/\\w+/kernel', name):
                critic_loss += l2_reg * l2_loss(var)
        return (critic_loss, actor_loss, td_error)

    def _translate_weights_to_torch(self, weights_dict, map_):
        if False:
            return 10
        model_dict = {map_[k]: convert_to_torch_tensor(np.transpose(v) if re.search('kernel', k) else v) for (k, v) in weights_dict.items() if re.search('default_policy/(actor_(hidden_0|out)|sequential(_1)?)/', k)}
        model_dict['policy_model.action_out_squashed.low_action'] = convert_to_torch_tensor(np.array([0.0]))
        model_dict['policy_model.action_out_squashed.action_range'] = convert_to_torch_tensor(np.array([1.0]))
        return model_dict
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))