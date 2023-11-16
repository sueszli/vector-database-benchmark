import os
import pathlib
import warnings
from copy import deepcopy
import numpy as np
import pytest
import torch as th
from stable_baselines3 import DDPG, DQN, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

def test_import_error():
    if False:
        print('Hello World!')
    with pytest.raises(ImportError) as excinfo:
        from stable_baselines3 import HER
        HER('MlpPolicy')
    assert 'documentation' in str(excinfo.value)

@pytest.mark.parametrize('model_class', [SAC, TD3, DDPG, DQN])
@pytest.mark.parametrize('image_obs_space', [True, False])
def test_her(model_class, image_obs_space):
    if False:
        while True:
            i = 10
    '\n    Test Hindsight Experience Replay.\n    '
    n_envs = 1
    n_bits = 4

    def env_fn():
        if False:
            print('Hello World!')
        return BitFlippingEnv(n_bits=n_bits, continuous=not model_class == DQN, image_obs_space=image_obs_space)
    env = make_vec_env(env_fn, n_envs)
    model = model_class('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=2, goal_selection_strategy='future', copy_info_dict=True), train_freq=4, gradient_steps=n_envs, policy_kwargs=dict(net_arch=[64]), learning_starts=100, buffer_size=int(20000.0))
    model.learn(total_timesteps=150)
    evaluate_policy(model, Monitor(env_fn()))

@pytest.mark.parametrize('model_class', [TD3, DQN])
@pytest.mark.parametrize('image_obs_space', [True, False])
def test_multiprocessing(model_class, image_obs_space):
    if False:
        print('Hello World!')

    def env_fn():
        if False:
            return 10
        return BitFlippingEnv(n_bits=4, continuous=not model_class == DQN, image_obs_space=image_obs_space)
    env = make_vec_env(env_fn, n_envs=2, vec_env_cls=SubprocVecEnv)
    model = model_class('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, buffer_size=int(20000.0), train_freq=4)
    model.learn(total_timesteps=150)

@pytest.mark.parametrize('goal_selection_strategy', ['final', 'episode', 'future', GoalSelectionStrategy.FINAL, GoalSelectionStrategy.EPISODE, GoalSelectionStrategy.FUTURE])
def test_goal_selection_strategy(goal_selection_strategy):
    if False:
        i = 10
        return i + 15
    '\n    Test different goal strategies.\n    '
    n_envs = 2

    def env_fn():
        if False:
            i = 10
            return i + 15
        return BitFlippingEnv(continuous=True)
    env = make_vec_env(env_fn, n_envs)
    normal_action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))
    model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(goal_selection_strategy=goal_selection_strategy, n_sampled_goal=2), train_freq=4, gradient_steps=n_envs, policy_kwargs=dict(net_arch=[64]), learning_starts=100, buffer_size=int(100000.0), action_noise=normal_action_noise)
    assert model.action_noise is not None
    model.learn(total_timesteps=150)

@pytest.mark.parametrize('model_class', [SAC, TD3, DDPG, DQN])
@pytest.mark.parametrize('use_sde', [False, True])
def test_save_load(tmp_path, model_class, use_sde):
    if False:
        i = 10
        return i + 15
    "\n    Test if 'save' and 'load' saves and loads model correctly\n    "
    if use_sde and model_class != SAC:
        pytest.skip('Only SAC has gSDE support')
    n_envs = 2
    n_bits = 4

    def env_fn():
        if False:
            print('Hello World!')
        return BitFlippingEnv(n_bits=n_bits, continuous=not model_class == DQN)
    env = make_vec_env(env_fn, n_envs)
    kwargs = dict(use_sde=True) if use_sde else {}
    model = model_class('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=2, goal_selection_strategy='future'), verbose=0, tau=0.05, batch_size=128, learning_rate=0.001, policy_kwargs=dict(net_arch=[64]), buffer_size=int(100000.0), gamma=0.98, gradient_steps=n_envs, train_freq=4, learning_starts=100, **kwargs)
    model.learn(total_timesteps=150)
    env.reset()
    action = np.array([env.action_space.sample() for _ in range(n_envs)])
    observations = env.step(action)[0]
    params = deepcopy(model.policy.state_dict())
    random_params = {param_name: th.rand_like(param) for (param_name, param) in params.items()}
    model.policy.load_state_dict(random_params)
    new_params = model.policy.state_dict()
    for k in params:
        assert not th.allclose(params[k], new_params[k]), 'Parameters did not change as expected.'
    params = new_params
    (selected_actions, _) = model.predict(observations, deterministic=True)
    model.save(tmp_path / 'test_save.zip')
    del model
    custom_objects = dict(learning_rate=2e-05, dummy=1.0)
    model_ = model_class.load(str(tmp_path / 'test_save.zip'), env=env, custom_objects=custom_objects, verbose=2)
    assert model_.verbose == 2
    assert model_.learning_rate == custom_objects['learning_rate']
    assert not hasattr(model_, 'dummy')
    model = model_class.load(str(tmp_path / 'test_save.zip'), env=env)
    new_params = model.policy.state_dict()
    for key in params:
        assert th.allclose(params[key], new_params[key]), 'Model parameters not the same after save and load.'
    (new_selected_actions, _) = model.predict(observations, deterministic=True)
    assert np.allclose(selected_actions, new_selected_actions, 0.0001)
    model.learn(total_timesteps=150)
    model = model_class.load(str(tmp_path / 'test_save.zip'), env=env, verbose=3, learning_rate=2.0)
    assert model.learning_rate == 2.0
    assert model.verbose == 3
    os.remove(tmp_path / 'test_save.zip')

@pytest.mark.parametrize('n_envs', [1, 2])
@pytest.mark.parametrize('truncate_last_trajectory', [False, True])
def test_save_load_replay_buffer(n_envs, tmp_path, recwarn, truncate_last_trajectory):
    if False:
        print('Hello World!')
    "\n    Test if 'save_replay_buffer' and 'load_replay_buffer' works correctly\n    "
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gym')
    path = pathlib.Path(tmp_path / 'replay_buffer.pkl')
    path.parent.mkdir(exist_ok=True, parents=True)

    def env_fn():
        if False:
            i = 10
            return i + 15
        return BitFlippingEnv(n_bits=4, continuous=True)
    env = make_vec_env(env_fn, n_envs)
    model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=2, goal_selection_strategy='future'), gradient_steps=n_envs, train_freq=4, buffer_size=int(20000.0), policy_kwargs=dict(net_arch=[64]), seed=0)
    model.learn(200)
    old_replay_buffer = deepcopy(model.replay_buffer)
    model.save_replay_buffer(path)
    del model.replay_buffer
    with pytest.raises(AttributeError):
        model.replay_buffer
    assert len(recwarn) == 0
    model.load_replay_buffer(path, truncate_last_traj=truncate_last_trajectory)
    if truncate_last_trajectory and (old_replay_buffer.dones[old_replay_buffer.pos - 1] == 0).any():
        assert len(recwarn) == 1
        warning = recwarn.pop(UserWarning)
        assert 'The last trajectory in the replay buffer will be truncated' in str(warning.message)
    else:
        assert len(recwarn) == 0
    replay_buffer = model.replay_buffer
    pos = replay_buffer.pos
    for key in ['observation', 'desired_goal', 'achieved_goal']:
        assert np.allclose(old_replay_buffer.observations[key][:pos], replay_buffer.observations[key][:pos])
        assert np.allclose(old_replay_buffer.next_observations[key][:pos], replay_buffer.next_observations[key][:pos])
    assert np.allclose(old_replay_buffer.actions[:pos], replay_buffer.actions[:pos])
    assert np.allclose(old_replay_buffer.rewards[:pos], replay_buffer.rewards[:pos])
    assert np.allclose(old_replay_buffer.dones[:pos - 1], replay_buffer.dones[:pos - 1])
    reset_num_timesteps = False if truncate_last_trajectory is False else True
    model.learn(200, reset_num_timesteps=reset_num_timesteps)

def test_full_replay_buffer():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if HER works correctly with a full replay buffer when using online sampling.\n    It should not sample the current episode which is not finished.\n    '
    n_bits = 4
    n_envs = 2

    def env_fn():
        if False:
            print('Hello World!')
        return BitFlippingEnv(n_bits=n_bits, continuous=True)
    env = make_vec_env(env_fn, n_envs)
    model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=2, goal_selection_strategy='future'), gradient_steps=1, train_freq=4, policy_kwargs=dict(net_arch=[64]), learning_starts=n_bits * n_envs, buffer_size=20 * n_envs, verbose=1, seed=757)
    model.learn(total_timesteps=100)

@pytest.mark.parametrize('n_envs', [1, 2])
@pytest.mark.parametrize('n_steps', [4, 5])
@pytest.mark.parametrize('handle_timeout_termination', [False, True])
def test_truncate_last_trajectory(n_envs, recwarn, n_steps, handle_timeout_termination):
    if False:
        print('Hello World!')
    "\n    Test if 'truncate_last_trajectory' works correctly\n    "
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gym')
    n_bits = 4

    def env_fn():
        if False:
            return 10
        return BitFlippingEnv(n_bits=n_bits, continuous=True)
    venv = make_vec_env(env_fn, n_envs)
    replay_buffer = HerReplayBuffer(buffer_size=int(10000.0), observation_space=venv.observation_space, action_space=venv.action_space, env=venv, n_envs=n_envs, n_sampled_goal=2, goal_selection_strategy='future')
    observations = venv.reset()
    for _ in range(n_steps):
        actions = np.random.rand(n_envs, n_bits)
        (next_observations, rewards, dones, infos) = venv.step(actions)
        replay_buffer.add(observations, next_observations, actions, rewards, dones, infos)
        observations = next_observations
    old_replay_buffer = deepcopy(replay_buffer)
    pos = replay_buffer.pos
    if handle_timeout_termination:
        env_idx_not_finished = np.where(replay_buffer._current_ep_start != pos)[0]
    assert len(recwarn) == 0
    replay_buffer.truncate_last_trajectory()
    if (old_replay_buffer.dones[pos - 1] == 0).any():
        assert len(recwarn) == 1
        warning = recwarn.pop(UserWarning)
        assert 'The last trajectory in the replay buffer will be truncated' in str(warning.message)
    else:
        assert len(recwarn) == 0
    assert (replay_buffer._current_ep_start == pos).all()
    assert (replay_buffer.dones[pos - 1] == 1).all()
    if handle_timeout_termination:
        assert (replay_buffer.timeouts[pos - 1, env_idx_not_finished] == 1).all()
    assert (replay_buffer.ep_length[pos - 1] != 0).all()
    for key in ['observation', 'desired_goal', 'achieved_goal']:
        assert np.allclose(old_replay_buffer.observations[key], replay_buffer.observations[key])
        assert np.allclose(old_replay_buffer.next_observations[key], replay_buffer.next_observations[key])
    assert np.allclose(old_replay_buffer.actions, replay_buffer.actions)
    assert np.allclose(old_replay_buffer.rewards, replay_buffer.rewards)
    assert np.allclose(old_replay_buffer.dones[:pos - 1], replay_buffer.dones[:pos - 1])
    assert np.allclose(old_replay_buffer.dones[pos:], replay_buffer.dones[pos:])
    for _ in range(10):
        actions = np.random.rand(n_envs, n_bits)
        (next_observations, rewards, dones, infos) = venv.step(actions)
        replay_buffer.add(observations, next_observations, actions, rewards, dones, infos)
        observations = next_observations
    for key in ['observation', 'desired_goal', 'achieved_goal']:
        assert np.allclose(old_replay_buffer.observations[key][:pos], replay_buffer.observations[key][:pos])
        assert np.allclose(old_replay_buffer.next_observations[key][:pos], replay_buffer.next_observations[key][:pos])
    assert np.allclose(old_replay_buffer.actions[:pos], replay_buffer.actions[:pos])
    assert np.allclose(old_replay_buffer.rewards[:pos], replay_buffer.rewards[:pos])
    assert np.allclose(old_replay_buffer.dones[:pos - 1], replay_buffer.dones[:pos - 1])
    end_pos = replay_buffer.pos
    for key in ['observation', 'desired_goal', 'achieved_goal']:
        assert not np.allclose(old_replay_buffer.observations[key][pos:end_pos], replay_buffer.observations[key][pos:end_pos])
        assert not np.allclose(old_replay_buffer.next_observations[key][pos:end_pos], replay_buffer.next_observations[key][pos:end_pos])
    assert not np.allclose(old_replay_buffer.actions[pos:end_pos], replay_buffer.actions[pos:end_pos])
    assert not np.allclose(old_replay_buffer.rewards[pos:end_pos], replay_buffer.rewards[pos:end_pos])
    assert not np.allclose(old_replay_buffer.dones[pos - 1:end_pos], replay_buffer.dones[pos - 1:end_pos])
    for key in ['observation', 'desired_goal', 'achieved_goal']:
        assert np.allclose(old_replay_buffer.observations[key][end_pos:], replay_buffer.observations[key][end_pos:])
        assert np.allclose(old_replay_buffer.next_observations[key][end_pos:], replay_buffer.next_observations[key][end_pos:])
    assert np.allclose(old_replay_buffer.actions[end_pos:], replay_buffer.actions[end_pos:])
    assert np.allclose(old_replay_buffer.rewards[end_pos:], replay_buffer.rewards[end_pos:])
    assert np.allclose(old_replay_buffer.dones[end_pos:], replay_buffer.dones[end_pos:])

@pytest.mark.parametrize('n_bits', [10])
def test_performance_her(n_bits):
    if False:
        for i in range(10):
            print('nop')
    '\n    That DQN+HER can solve BitFlippingEnv.\n    It should not work when n_sampled_goal=0 (DQN alone).\n    '
    n_envs = 2

    def env_fn():
        if False:
            for i in range(10):
                print('nop')
        return BitFlippingEnv(n_bits=n_bits, continuous=False)
    env = make_vec_env(env_fn, n_envs)
    model = DQN('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=5, goal_selection_strategy='future'), verbose=1, learning_rate=0.0005, train_freq=1, gradient_steps=n_envs, learning_starts=100, exploration_final_eps=0.02, target_update_interval=500, seed=0, batch_size=32, buffer_size=int(100000.0))
    model.learn(total_timesteps=5000, log_interval=50)
    assert np.mean(model.ep_success_buffer) > 0.9