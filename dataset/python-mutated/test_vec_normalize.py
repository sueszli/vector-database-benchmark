import operator
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import SAC, TD3, HerReplayBuffer
from stable_baselines3.common.envs import FakeImageEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, sync_envs_normalization, unwrap_vec_normalize
ENV_ID = 'Pendulum-v1'

class DummyRewardEnv(gym.Env):
    metadata: Dict[str, Any] = {}

    def __init__(self, return_reward_idx=0):
        if False:
            print('Hello World!')
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]))
        self.returned_rewards = [0, 1, 3, 4]
        self.return_reward_idx = return_reward_idx
        self.t = self.return_reward_idx

    def step(self, action):
        if False:
            while True:
                i = 10
        self.t += 1
        index = (self.t + self.return_reward_idx) % len(self.returned_rewards)
        returned_value = self.returned_rewards[index]
        terminated = False
        truncated = self.t == len(self.returned_rewards)
        return (np.array([returned_value]), returned_value, terminated, truncated, {})

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict]=None):
        if False:
            print('Hello World!')
        if seed is not None:
            super().reset(seed=seed)
        self.t = 0
        return (np.array([self.returned_rewards[self.return_reward_idx]]), {})

class DummyDictEnv(gym.Env):
    """
    Dummy gym goal env for testing purposes
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.observation_space = spaces.Dict({'observation': spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32), 'achieved_goal': spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32), 'desired_goal': spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32)})
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict]=None):
        if False:
            for i in range(10):
                print('nop')
        if seed is not None:
            super().reset(seed=seed)
        return (self.observation_space.sample(), {})

    def step(self, action):
        if False:
            i = 10
            return i + 15
        obs = self.observation_space.sample()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        terminated = np.random.rand() > 0.8
        return (obs, reward, terminated, False, {})

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, _info) -> np.float32:
        if False:
            print('Hello World!')
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > 0).astype(np.float32)

class DummyMixedDictEnv(gym.Env):
    """
    Dummy mixed gym env for testing purposes
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.observation_space = spaces.Dict({'obs1': spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32), 'obs2': spaces.Discrete(1), 'obs3': spaces.Box(low=-20.0, high=20.0, shape=(4,), dtype=np.float32)})
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict]=None):
        if False:
            for i in range(10):
                print('nop')
        if seed is not None:
            super().reset(seed=seed)
        return (self.observation_space.sample(), {})

    def step(self, action):
        if False:
            i = 10
            return i + 15
        obs = self.observation_space.sample()
        terminated = np.random.rand() > 0.8
        return (obs, 0.0, terminated, False, {})

def allclose(obs_1, obs_2):
    if False:
        while True:
            i = 10
    '\n    Generalized np.allclose() to work with dict spaces.\n    '
    if isinstance(obs_1, dict):
        all_close = True
        for key in obs_1.keys():
            if not np.allclose(obs_1[key], obs_2[key]):
                all_close = False
                break
        return all_close
    return np.allclose(obs_1, obs_2)

def make_env():
    if False:
        while True:
            i = 10
    return Monitor(gym.make(ENV_ID))

def make_env_render():
    if False:
        print('Hello World!')
    return Monitor(gym.make(ENV_ID, render_mode='rgb_array'))

def make_dict_env():
    if False:
        i = 10
        return i + 15
    return Monitor(DummyDictEnv())

def make_image_env():
    if False:
        for i in range(10):
            print('nop')
    return Monitor(FakeImageEnv())

def check_rms_equal(rmsa, rmsb):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(rmsa, dict):
        for key in rmsa.keys():
            assert np.all(rmsa[key].mean == rmsb[key].mean)
            assert np.all(rmsa[key].var == rmsb[key].var)
            assert np.all(rmsa[key].count == rmsb[key].count)
    else:
        assert np.all(rmsa.mean == rmsb.mean)
        assert np.all(rmsa.var == rmsb.var)
        assert np.all(rmsa.count == rmsb.count)

def check_vec_norm_equal(norma, normb):
    if False:
        for i in range(10):
            print('nop')
    assert norma.observation_space == normb.observation_space
    assert norma.action_space == normb.action_space
    assert norma.num_envs == normb.num_envs
    check_rms_equal(norma.obs_rms, normb.obs_rms)
    check_rms_equal(norma.ret_rms, normb.ret_rms)
    assert norma.clip_obs == normb.clip_obs
    assert norma.clip_reward == normb.clip_reward
    assert norma.norm_obs == normb.norm_obs
    assert norma.norm_reward == normb.norm_reward
    assert np.all(norma.returns == normb.returns)
    assert norma.gamma == normb.gamma
    assert norma.epsilon == normb.epsilon
    assert norma.training == normb.training

def _make_warmstart(env_fn, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Warm-start VecNormalize by stepping through 100 actions.'
    venv = DummyVecEnv([env_fn])
    venv = VecNormalize(venv, **kwargs)
    venv.reset()
    venv.get_original_obs()
    for _ in range(100):
        actions = [venv.action_space.sample()]
        venv.step(actions)
    return venv

def _make_warmstart_cliffwalking(**kwargs):
    if False:
        i = 10
        return i + 15
    'Warm-start VecNormalize by stepping through CliffWalking'
    return _make_warmstart(lambda : gym.make('CliffWalking-v0'), **kwargs)

def _make_warmstart_cartpole():
    if False:
        for i in range(10):
            print('nop')
    'Warm-start VecNormalize by stepping through CartPole'
    return _make_warmstart(lambda : gym.make('CartPole-v1'))

def _make_warmstart_dict_env(**kwargs):
    if False:
        print('Hello World!')
    'Warm-start VecNormalize by stepping through DummyDictEnv'
    return _make_warmstart(make_dict_env, **kwargs)

def test_runningmeanstd():
    if False:
        return 10
    'Test RunningMeanStd object'
    for (x_1, x_2, x_3) in [(np.random.randn(3), np.random.randn(4), np.random.randn(5)), (np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2))]:
        rms = RunningMeanStd(epsilon=0.0, shape=x_1.shape[1:])
        x_cat = np.concatenate([x_1, x_2, x_3], axis=0)
        moments_1 = [x_cat.mean(axis=0), x_cat.var(axis=0)]
        rms.update(x_1)
        rms.update(x_2)
        rms.update(x_3)
        moments_2 = [rms.mean, rms.var]
        assert np.allclose(moments_1, moments_2)

def test_combining_stats():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(4)
    for shape in [(1,), (3,), (3, 4)]:
        values = []
        rms_1 = RunningMeanStd(shape=shape)
        rms_2 = RunningMeanStd(shape=shape)
        rms_3 = RunningMeanStd(shape=shape)
        for _ in range(15):
            value = np.random.randn(*shape)
            rms_1.update(value)
            rms_3.update(value)
            values.append(value)
        for _ in range(19):
            value = np.random.randn(*shape) + 1.0
            rms_2.update(value)
            rms_3.update(value)
            values.append(value)
        rms_1.combine(rms_2)
        assert np.allclose(rms_3.mean, rms_1.mean)
        assert np.allclose(rms_3.var, rms_1.var)
        rms_4 = rms_3.copy()
        assert np.allclose(rms_4.mean, rms_3.mean)
        assert np.allclose(rms_4.var, rms_3.var)
        assert np.allclose(rms_4.count, rms_3.count)
        assert id(rms_4.mean) != id(rms_3.mean)
        assert id(rms_4.var) != id(rms_3.var)
        x_cat = np.concatenate(values, axis=0)
        assert np.allclose(x_cat.mean(axis=0), rms_4.mean)
        assert np.allclose(x_cat.var(axis=0), rms_4.var)

def test_obs_rms_vec_normalize():
    if False:
        while True:
            i = 10
    env_fns = [lambda : DummyRewardEnv(0), lambda : DummyRewardEnv(1)]
    env = DummyVecEnv(env_fns)
    env = VecNormalize(env)
    env.reset()
    assert np.allclose(env.obs_rms.mean, 0.5, atol=0.0001)
    assert np.allclose(env.ret_rms.mean, 0.0, atol=0.0001)
    env.step([env.action_space.sample() for _ in range(len(env_fns))])
    assert np.allclose(env.obs_rms.mean, 1.25, atol=0.0001)
    assert np.allclose(env.ret_rms.mean, 2, atol=0.0001)
    for _ in range(3000):
        env.step([env.action_space.sample() for _ in range(len(env_fns))])
    assert np.allclose(env.obs_rms.mean, 2.0, atol=0.001)
    assert np.allclose(env.ret_rms.mean, 5.688, atol=0.001)

@pytest.mark.parametrize('make_gym_env', [make_env, make_dict_env, make_image_env])
def test_vec_env(tmp_path, make_gym_env):
    if False:
        print('Hello World!')
    'Test VecNormalize Object'
    clip_obs = 0.5
    clip_reward = 5.0
    orig_venv = DummyVecEnv([make_gym_env])
    norm_venv = VecNormalize(orig_venv, norm_obs=True, norm_reward=True, clip_obs=clip_obs, clip_reward=clip_reward)
    assert orig_venv.render_mode is None
    assert norm_venv.render_mode is None
    (_, done) = (norm_venv.reset(), [False])
    while not done[0]:
        actions = [norm_venv.action_space.sample()]
        (obs, rew, done, _) = norm_venv.step(actions)
        if isinstance(obs, dict):
            for key in obs.keys():
                assert np.max(np.abs(obs[key])) <= clip_obs
        else:
            assert np.max(np.abs(obs)) <= clip_obs
        assert np.max(np.abs(rew)) <= clip_reward
    path = tmp_path / 'vec_normalize'
    norm_venv.save(path)
    assert orig_venv.render_mode is None
    deserialized = VecNormalize.load(path, venv=orig_venv)
    assert deserialized.render_mode is None
    check_vec_norm_equal(norm_venv, deserialized)
    vec_env = DummyVecEnv([make_env_render])
    assert vec_env.render_mode == 'rgb_array'
    if make_gym_env == make_env:
        assert VecNormalize.load(path, venv=vec_env).render_mode == 'rgb_array'
        assert VecNormalize(vec_env).render_mode == 'rgb_array'

def test_get_original():
    if False:
        print('Hello World!')
    venv = _make_warmstart_cartpole()
    for _ in range(3):
        actions = [venv.action_space.sample()]
        (obs, rewards, _, _) = venv.step(actions)
        obs = obs[0]
        orig_obs = venv.get_original_obs()[0]
        rewards = rewards[0]
        orig_rewards = venv.get_original_reward()[0]
        assert np.all(orig_rewards == 1)
        assert orig_obs.shape == obs.shape
        assert orig_rewards.dtype == rewards.dtype
        assert not np.array_equal(orig_obs, obs)
        assert not np.array_equal(orig_rewards, rewards)
        np.testing.assert_allclose(venv.normalize_obs(orig_obs), obs)
        np.testing.assert_allclose(venv.normalize_reward(orig_rewards), rewards)

def test_get_original_dict():
    if False:
        i = 10
        return i + 15
    venv = _make_warmstart_dict_env()
    for _ in range(3):
        actions = [venv.action_space.sample()]
        (obs, rewards, _, _) = venv.step(actions)
        orig_obs = venv.get_original_obs()
        rewards = rewards[0]
        orig_rewards = venv.get_original_reward()[0]
        for key in orig_obs.keys():
            assert orig_obs[key].shape == obs[key].shape
        assert orig_rewards.dtype == rewards.dtype
        assert not allclose(orig_obs, obs)
        assert not np.array_equal(orig_rewards, rewards)
        assert allclose(venv.normalize_obs(orig_obs), obs)
        np.testing.assert_allclose(venv.normalize_reward(orig_rewards), rewards)

def test_normalize_external():
    if False:
        i = 10
        return i + 15
    venv = _make_warmstart_cartpole()
    rewards = np.array([1, 1])
    norm_rewards = venv.normalize_reward(rewards)
    assert norm_rewards.shape == rewards.shape
    assert np.all(norm_rewards < 1)

def test_normalize_dict_selected_keys():
    if False:
        while True:
            i = 10
    venv = _make_warmstart_dict_env(norm_obs=True, norm_obs_keys=['observation'])
    for _ in range(3):
        actions = [venv.action_space.sample()]
        (obs, rewards, _, _) = venv.step(actions)
        orig_obs = venv.get_original_obs()
        np.testing.assert_array_compare(operator.__ne__, obs['observation'], orig_obs['observation'])
        assert allclose(venv.normalize_obs(orig_obs), obs)
        np.testing.assert_array_equal(obs['achieved_goal'], orig_obs['achieved_goal'])

def test_her_normalization():
    if False:
        for i in range(10):
            print('nop')
    env = DummyVecEnv([make_dict_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    eval_env = DummyVecEnv([make_dict_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)
    model = SAC('MultiInputPolicy', env, verbose=1, learning_starts=100, policy_kwargs=dict(net_arch=[64]), replay_buffer_kwargs=dict(n_sampled_goal=2), replay_buffer_class=HerReplayBuffer, seed=2)
    assert model.get_vec_normalize_env() is env
    model.set_env(eval_env)
    assert model.get_vec_normalize_env() is eval_env
    model.learn(total_timesteps=10)
    model.set_env(env)
    model.learn(total_timesteps=150)
    assert isinstance(model.get_vec_normalize_env(), VecNormalize)

@pytest.mark.parametrize('model_class', [SAC, TD3])
def test_offpolicy_normalization(model_class):
    if False:
        while True:
            i = 10
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)
    model = model_class('MlpPolicy', env, verbose=1, learning_starts=100, policy_kwargs=dict(net_arch=[64]))
    assert model.get_vec_normalize_env() is env
    model.set_env(eval_env)
    assert model.get_vec_normalize_env() is eval_env
    model.learn(total_timesteps=10)
    model.set_env(env)
    model.learn(total_timesteps=150)
    assert isinstance(model.get_vec_normalize_env(), VecNormalize)

@pytest.mark.parametrize('make_env', [make_env, make_dict_env])
def test_sync_vec_normalize(make_env):
    if False:
        print('Hello World!')
    original_env = DummyVecEnv([make_env])
    assert unwrap_vec_normalize(original_env) is None
    env = VecNormalize(original_env, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=100.0)
    assert isinstance(unwrap_vec_normalize(env), VecNormalize)
    if not isinstance(env.observation_space, spaces.Dict):
        env = VecFrameStack(env, 1)
        assert isinstance(unwrap_vec_normalize(env), VecNormalize)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=100.0)
    if not isinstance(env.observation_space, spaces.Dict):
        eval_env = VecFrameStack(eval_env, 1)
    env.seed(0)
    env.action_space.seed(0)
    env.reset()
    latest_reward = None
    for _ in range(100):
        (_, latest_reward, _, _) = env.step([env.action_space.sample()])
    original_latest_reward = env.get_original_reward()
    assert np.allclose(original_latest_reward, env.unnormalize_reward(latest_reward))
    obs = env.reset()
    dummy_rewards = np.random.rand(10)
    original_obs = env.get_original_obs()
    assert allclose(original_obs, env.unnormalize_obs(obs))
    assert not allclose(obs, eval_env.normalize_obs(original_obs))
    sync_envs_normalization(env, eval_env)
    assert allclose(obs, eval_env.normalize_obs(original_obs))
    assert allclose(env.normalize_reward(dummy_rewards), eval_env.normalize_reward(dummy_rewards))
    env = VecNormalize(original_env, norm_obs=False, norm_reward=True, clip_reward=100.0)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False)
    env.reset()
    env.step([env.action_space.sample()])
    assert not np.allclose(env.ret_rms.mean, eval_env.ret_rms.mean)
    sync_envs_normalization(env, eval_env)
    assert np.allclose(env.ret_rms.mean, eval_env.ret_rms.mean)
    assert np.allclose(env.ret_rms.var, eval_env.ret_rms.var)

def test_discrete_obs():
    if False:
        return 10
    with pytest.raises(ValueError, match='.*only supports.*'):
        _make_warmstart_cliffwalking()
    _make_warmstart_cliffwalking(norm_obs=False)

def test_non_dict_obs_keys():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='.*is applicable only.*'):
        _make_warmstart(lambda : DummyRewardEnv(), norm_obs_keys=['key'])
    with pytest.raises(ValueError, match='.* explicitely pass the observation keys.*'):
        _make_warmstart(lambda : DummyMixedDictEnv())
    _make_warmstart(lambda : DummyMixedDictEnv(), norm_obs_keys=['obs1', 'obs3'])
    _make_warmstart(lambda : DummyMixedDictEnv(), norm_obs=False)