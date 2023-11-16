"""Env wrappers
Note that this file is adapted from `https://pypi.org/project/gym-vec-env` and
`https://github.com/openai/baselines/blob/master/baselines/common/*wrappers.py`
"""
from collections import deque
from functools import partial
from multiprocessing import Pipe, Process, cpu_count
from sys import platform
import cv2
import gym
import numpy as np
from gym import spaces
__all__ = ('build_env', 'TimeLimit', 'NoopResetEnv', 'FireResetEnv', 'EpisodicLifeEnv', 'MaxAndSkipEnv', 'ClipRewardEnv', 'WarpFrame', 'FrameStack', 'LazyFrames', 'RewardScaler', 'SubprocVecEnv', 'VecFrameStack', 'Monitor')
cv2.ocl.setUseOpenCL(False)
id2type = dict()
for _env in gym.envs.registry.all():
    id2type[_env.id] = _env._entry_point.split(':')[0].rsplit('.', 1)[1]

def build_env(env_id, vectorized=False, seed=0, reward_scale=1.0, nenv=0):
    if False:
        for i in range(10):
            print('nop')
    'Build env based on options'
    env_type = id2type[env_id]
    nenv = nenv or cpu_count() // (1 + (platform == 'darwin'))
    stack = env_type == 'atari'
    if not vectorized:
        env = _make_env(env_id, env_type, seed, reward_scale, stack)
    else:
        env = _make_vec_env(env_id, env_type, nenv, seed, reward_scale, stack)
    return env

def _make_env(env_id, env_type, seed, reward_scale, frame_stack=True):
    if False:
        for i in range(10):
            print('nop')
    'Make single env'
    if env_type == 'atari':
        env = gym.make(env_id)
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = Monitor(env)
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        if frame_stack:
            env = FrameStack(env, 4)
    elif env_type == 'classic_control':
        env = Monitor(gym.make(env_id))
    else:
        raise NotImplementedError
    if reward_scale != 1:
        env = RewardScaler(env, reward_scale)
    env.seed(seed)
    return env

def _make_vec_env(env_id, env_type, nenv, seed, reward_scale, frame_stack=True):
    if False:
        return 10
    'Make vectorized env'
    env = SubprocVecEnv([partial(_make_env, env_id, env_type, seed + i, reward_scale, False) for i in range(nenv)])
    if frame_stack:
        env = VecFrameStack(env, 4)
    return env

class TimeLimit(gym.Wrapper):

    def __init__(self, env, max_episode_steps=None):
        if False:
            return 10
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        if False:
            return 10
        (observation, reward, done, info) = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return (observation, reward, done, info)

    def reset(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class NoopResetEnv(gym.Wrapper):

    def __init__(self, env, noop_max=30):
        if False:
            print('Hello World!')
        'Sample initial states by taking random number of no-ops on reset.\n        No-op is assumed to be action 0.\n        '
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Do no-op action for a number of steps in [1, noop_max].'
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            (obs, _, done, _) = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        if False:
            while True:
                i = 10
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):

    def __init__(self, env):
        if False:
            i = 10
            return i + 15
        'Take action on reset for environments that are fixed until firing.'
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        if False:
            return 10
        self.env.reset(**kwargs)
        (obs, _, done, _) = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        (obs, _, done, _) = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        if False:
            print('Hello World!')
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env):
        if False:
            for i in range(10):
                print('nop')
        'Make end-of-life == end-of-episode, but only reset on true game over.\n        Done by DeepMind for the DQN and co. since it helps value estimation.\n        '
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        if False:
            i = 10
            return i + 15
        (obs, reward, done, info) = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            done = True
        self.lives = lives
        return (obs, reward, done, info)

    def reset(self, **kwargs):
        if False:
            print('Hello World!')
        'Reset only when lives are exhausted.\n        This way all states are still reachable even though lives are episodic,\n        and the learner need not know about any of this behind-the-scenes.\n        '
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            (obs, _, _, _) = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):

    def __init__(self, env, skip=4):
        if False:
            i = 10
            return i + 15
        'Return only every `skip`-th frame'
        super(MaxAndSkipEnv, self).__init__(env)
        shape = (2,) + env.observation_space.shape
        self._obs_buffer = np.zeros(shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        if False:
            return 10
        'Repeat action, sum reward, and max over last observations.'
        total_reward = 0.0
        done = info = None
        for i in range(self._skip):
            (obs, reward, done, info) = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return (max_frame, total_reward, done, info)

    def reset(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):

    def __init__(self, env):
        if False:
            i = 10
            return i + 15
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        if False:
            print('Hello World!')
        'Bin reward to {+1, 0, -1} by its sign.'
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):

    def __init__(self, env, width=84, height=84, grayscale=True):
        if False:
            print('Hello World!')
        'Warp frames to 84x84 as done in the Nature paper and later work.'
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        shape = (self.height, self.width, 1 if self.grayscale else 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, frame):
        if False:
            return 10
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        size = (self.width, self.height)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame

class FrameStack(gym.Wrapper):

    def __init__(self, env, k):
        if False:
            i = 10
            return i + 15
        'Stack k last frames.\n        Returns lazy array, which is much more memory efficient.\n        See Also `LazyFrames`\n        '
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shape = shp[:-1] + (shp[-1] * k,)
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=env.observation_space.dtype)

    def reset(self):
        if False:
            i = 10
            return i + 15
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.asarray(self._get_ob())

    def step(self, action):
        if False:
            while True:
                i = 10
        (ob, reward, done, info) = self.env.step(action)
        self.frames.append(ob)
        return (np.asarray(self._get_ob()), reward, done, info)

    def _get_ob(self):
        if False:
            return 10
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):

    def __init__(self, frames):
        if False:
            for i in range(10):
                print('nop')
        "This object ensures that common frames between the observations are\n        only stored once. It exists purely to optimize memory usage which can be\n        huge for DQN's 1M frames replay buffers.\n\n        This object should only be converted to numpy array before being passed\n        to the model. You'd not believe how complex the previous solution was.\n        "
        self._frames = frames
        self._out = None

    def _force(self):
        if False:
            for i in range(10):
                print('nop')
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        if False:
            print('Hello World!')
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._force())

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        return self._force()[i]

class RewardScaler(gym.RewardWrapper):
    """Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance drastically.
    """

    def __init__(self, env, scale=0.01):
        if False:
            while True:
                i = 10
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        if False:
            while True:
                i = 10
        return reward * self.scale

class VecFrameStack(object):

    def __init__(self, env, k):
        if False:
            return 10
        self.env = env
        self.k = k
        self.action_space = env.action_space
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shape = shp[:-1] + (shp[-1] * k,)
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=env.observation_space.dtype)

    def reset(self):
        if False:
            print('Hello World!')
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.asarray(self._get_ob())

    def step(self, action):
        if False:
            i = 10
            return i + 15
        (ob, reward, done, info) = self.env.step(action)
        self.frames.append(ob)
        return (np.asarray(self._get_ob()), reward, done, info)

    def _get_ob(self):
        if False:
            return 10
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

def _worker(remote, parent_remote, env_fn_wrapper):
    if False:
        for i in range(10):
            print('nop')
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        (cmd, data) = remote.recv()
        if cmd == 'step':
            (ob, reward, done, info) = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env._reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents
    """

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.x = x

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        if False:
            for i in range(10):
                print('nop')
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(object):

    def __init__(self, env_fns):
        if False:
            print('Hello World!')
        '\n        envs: list of gym environments to run in subprocesses\n        '
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        (self.remotes, self.work_remotes) = zip(*[Pipe() for _ in range(nenvs)])
        zipped_args = zip(self.work_remotes, self.remotes, env_fns)
        self.ps = [Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))) for (work_remote, remote, env_fn) in zipped_args]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        (observation_space, action_space) = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def _step_async(self, actions):
        if False:
            print('Hello World!')
        '\n            Tell all the environments to start taking a step\n            with the given actions.\n            Call step_wait() to get the results of the step.\n            You should not call this if a step_async run is\n            already pending.\n            '
        for (remote, action) in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def _step_wait(self):
        if False:
            while True:
                i = 10
        '\n            Wait for the step taken with step_async().\n            Returns (obs, rews, dones, infos):\n             - obs: an array of observations, or a tuple of\n                    arrays of observations.\n             - rews: an array of rewards\n             - dones: an array of "episode done" booleans\n             - infos: a sequence of info objects\n            '
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        (obs, rews, dones, infos) = zip(*results)
        return (np.stack(obs), np.stack(rews), np.stack(dones), infos)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        '\n            Reset all the environments and return an array of\n            observations, or a tuple of observation arrays.\n            If step_async is still doing work, that work will\n            be cancelled and step_wait() should not be called\n            until step_async() is invoked again.\n            '
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def _reset_task(self):
        if False:
            i = 10
            return i + 15
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if False:
            i = 10
            return i + 15
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.nenvs

    def step(self, actions):
        if False:
            print('Hello World!')
        self._step_async(actions)
        return self._step_wait()

class Monitor(gym.Wrapper):

    def __init__(self, env):
        if False:
            i = 10
            return i + 15
        super(Monitor, self).__init__(env)
        self._monitor_rewards = None

    def reset(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._monitor_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        if False:
            while True:
                i = 10
        (o_, r, done, info) = self.env.step(action)
        self._monitor_rewards.append(r)
        if done:
            info['episode'] = {'r': sum(self._monitor_rewards), 'l': len(self._monitor_rewards)}
        return (o_, r, done, info)

class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        if False:
            return 10
        low = self.action_space.low
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action

    def _reverse_action(self, action):
        if False:
            while True:
                i = 10
        low = self.action_space.low
        high = self.action_space.high
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        return action

def unit_test():
    if False:
        i = 10
        return i + 15
    env_id = 'CartPole-v0'
    unwrapped_env = gym.make(env_id)
    wrapped_env = build_env(env_id, False)
    o = wrapped_env.reset()
    print('Reset {} observation shape {}'.format(env_id, o.shape))
    done = False
    while not done:
        a = unwrapped_env.action_space.sample()
        (o_, r, done, info) = wrapped_env.step(a)
        print('Take action {} get reward {} info {}'.format(a, r, info))
    env_id = 'PongNoFrameskip-v4'
    nenv = 2
    unwrapped_env = gym.make(env_id)
    wrapped_env = build_env(env_id, True, nenv=nenv)
    o = wrapped_env.reset()
    print('Reset {} observation shape {}'.format(env_id, o.shape))
    for _ in range(1000):
        a = [unwrapped_env.action_space.sample() for _ in range(nenv)]
        a = np.asarray(a, 'int64')
        (o_, r, done, info) = wrapped_env.step(a)
        print('Take action {} get reward {} info {}'.format(a, r, info))
if __name__ == '__main__':
    unit_test()