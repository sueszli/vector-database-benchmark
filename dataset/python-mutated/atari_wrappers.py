from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Union
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.images import rgb2gray, resize

@PublicAPI
def is_atari(env: Union[gym.Env, str]) -> bool:
    if False:
        return 10
    'Returns, whether a given env object or env descriptor (str) is an Atari env.\n\n    Args:\n        env: The gym.Env object or a string descriptor of the env (e.g. "ALE/Pong-v5").\n\n    Returns:\n        Whether `env` is an Atari environment.\n    '
    if not isinstance(env, str):
        if hasattr(env.observation_space, 'shape') and env.observation_space.shape is not None and (len(env.observation_space.shape) <= 2):
            return False
        return 'AtariEnv<ALE' in str(env)
    else:
        return env.startswith('ALE/')

@PublicAPI
def get_wrapper_by_cls(env, cls):
    if False:
        print('Hello World!')
    'Returns the gym env wrapper of the given class, or None.'
    currentenv = env
    while True:
        if isinstance(currentenv, cls):
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            return None

@PublicAPI
class MonitorEnv(gym.Wrapper):

    def __init__(self, env=None):
        if False:
            i = 10
            return i + 15
        'Record episodes stats prior to EpisodicLifeEnv, etc.'
        gym.Wrapper.__init__(self, env)
        self._current_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._num_episodes = 0
        self._num_returned = 0

    def reset(self, **kwargs):
        if False:
            i = 10
            return i + 15
        (obs, info) = self.env.reset(**kwargs)
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)
        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1
        self._current_reward = 0
        self._num_steps = 0
        return (obs, info)

    def step(self, action):
        if False:
            while True:
                i = 10
        (obs, rew, terminated, truncated, info) = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        return (obs, rew, terminated, truncated, info)

    def get_episode_rewards(self):
        if False:
            print('Hello World!')
        return self._episode_rewards

    def get_episode_lengths(self):
        if False:
            while True:
                i = 10
        return self._episode_lengths

    def get_total_steps(self):
        if False:
            return 10
        return self._total_steps

    def next_episode_results(self):
        if False:
            i = 10
            return i + 15
        for i in range(self._num_returned, len(self._episode_rewards)):
            yield (self._episode_rewards[i], self._episode_lengths[i])
        self._num_returned = len(self._episode_rewards)

@PublicAPI
class NoopResetEnv(gym.Wrapper):

    def __init__(self, env, noop_max=30):
        if False:
            i = 10
            return i + 15
        'Sample initial states by taking random number of no-ops on reset.\n        No-op is assumed to be action 0.\n        '
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Do no-op action for a number of steps in [1, noop_max].'
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            try:
                noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
            except AttributeError:
                noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            (obs, _, terminated, truncated, info) = self.env.step(self.noop_action)
            if terminated or truncated:
                (obs, info) = self.env.reset(**kwargs)
        return (obs, info)

    def step(self, ac):
        if False:
            for i in range(10):
                print('nop')
        return self.env.step(ac)

@PublicAPI
class ClipRewardEnv(gym.RewardWrapper):

    def __init__(self, env):
        if False:
            while True:
                i = 10
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        if False:
            print('Hello World!')
        'Bin reward to {+1, 0, -1} by its sign.'
        return np.sign(reward)

@PublicAPI
class FireResetEnv(gym.Wrapper):

    def __init__(self, env):
        if False:
            i = 10
            return i + 15
        'Take action on reset.\n\n        For environments that are fixed until firing.'
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        if False:
            print('Hello World!')
        self.env.reset(**kwargs)
        (obs, _, terminated, truncated, _) = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        (obs, _, terminated, truncated, info) = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return (obs, info)

    def step(self, ac):
        if False:
            i = 10
            return i + 15
        return self.env.step(ac)

@PublicAPI
class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env):
        if False:
            print('Hello World!')
        'Make end-of-life == end-of-episode, but only reset on true game over.\n        Done by DeepMind for the DQN and co. since it helps value estimation.\n        '
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_terminated = True

    def step(self, action):
        if False:
            i = 10
            return i + 15
        (obs, reward, terminated, truncated, info) = self.env.step(action)
        self.was_real_terminated = terminated
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return (obs, reward, terminated, truncated, info)

    def reset(self, **kwargs):
        if False:
            return 10
        'Reset only when lives are exhausted.\n        This way all states are still reachable even though lives are episodic,\n        and the learner need not know about any of this behind-the-scenes.\n        '
        if self.was_real_terminated:
            (obs, info) = self.env.reset(**kwargs)
        else:
            (obs, _, _, _, info) = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return (obs, info)

@PublicAPI
class MaxAndSkipEnv(gym.Wrapper):

    def __init__(self, env, skip=4):
        if False:
            i = 10
            return i + 15
        'Return only every `skip`-th frame'
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        if False:
            print('Hello World!')
        'Repeat action, sum reward, and max over last observations.'
        total_reward = 0.0
        terminated = truncated = info = None
        for i in range(self._skip):
            (obs, reward, terminated, truncated, info) = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return (max_frame, total_reward, terminated, truncated, info)

    def reset(self, **kwargs):
        if False:
            return 10
        return self.env.reset(**kwargs)

@PublicAPI
class WarpFrame(gym.ObservationWrapper):

    def __init__(self, env, dim):
        if False:
            print('Hello World!')
        'Warp frames to the specified size (dim x dim).'
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        if False:
            while True:
                i = 10
        frame = rgb2gray(frame)
        frame = resize(frame, height=self.height, width=self.width)
        return frame[:, :, None]

@PublicAPI
class FrameStack(gym.Wrapper):

    def __init__(self, env, k):
        if False:
            for i in range(10):
                print('nop')
        'Stack k last frames.'
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self, *, seed=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        (ob, infos) = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(ob)
        return (self._get_ob(), infos)

    def step(self, action):
        if False:
            i = 10
            return i + 15
        (ob, reward, terminated, truncated, info) = self.env.step(action)
        self.frames.append(ob)
        return (self._get_ob(), reward, terminated, truncated, info)

    def _get_ob(self):
        if False:
            while True:
                i = 10
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

@PublicAPI
class FrameStackTrajectoryView(gym.ObservationWrapper):

    def __init__(self, env):
        if False:
            return 10
        'No stacking. Trajectory View API takes care of this.'
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        assert shp[2] == 1
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1]), dtype=env.observation_space.dtype)

    def observation(self, observation):
        if False:
            print('Hello World!')
        return np.squeeze(observation, axis=-1)

@PublicAPI
class ScaledFloatFrame(gym.ObservationWrapper):

    def __init__(self, env):
        if False:
            for i in range(10):
                print('nop')
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        if False:
            return 10
        return np.array(observation).astype(np.float32) / 255.0

@PublicAPI
def wrap_deepmind(env, dim=84, framestack=True, noframeskip=False):
    if False:
        return 10
    'Configure environment for DeepMind-style Atari.\n\n    Note that we assume reward clipping is done outside the wrapper.\n\n    Args:\n        env: The env object to wrap.\n        dim: Dimension to resize observations to (dim x dim).\n        framestack: Whether to framestack observations.\n    '
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if env.spec is not None and noframeskip is True:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, dim)
    if framestack is True:
        env = FrameStack(env, 4)
    return env