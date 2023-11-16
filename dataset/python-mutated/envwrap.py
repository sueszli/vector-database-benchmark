from builtins import object
try:
    import roboschool
except:
    pass
import gym
import numpy as np
from config import config
MAX_FRAMES = config['env']['max_frames']
gym.logger.level = 40

def get_env(env_name, *args, **kwargs):
    if False:
        return 10
    MAPPING = {'CartPole-v0': CartPoleWrapper}
    if env_name in MAPPING:
        return MAPPING[env_name](env_name, *args, **kwargs)
    else:
        return NoTimeLimitMujocoWrapper(env_name, *args, **kwargs)

class GymWrapper(object):
    """
  Generic wrapper for OpenAI gym environments.
  """

    def __init__(self, env_name):
        if False:
            while True:
                i = 10
        self.internal_env = gym.make(env_name)
        self.observation_space = self.internal_env.observation_space
        self.action_space = self.internal_env.action_space
        self.custom_init()

    def custom_init(self):
        if False:
            while True:
                i = 10
        pass

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.clock = 0
        return self.preprocess_obs(self.internal_env.reset())

    def sample(self):
        if False:
            while True:
                i = 10
        return self.action_space.sample()

    def normalize_actions(self, actions):
        if False:
            while True:
                i = 10
        return actions

    def unnormalize_actions(self, actions):
        if False:
            return 10
        return actions

    def preprocess_obs(self, obs):
        if False:
            return 10
        return obs

    def step(self, normalized_action):
        if False:
            return 10
        out = self.internal_env.step(normalized_action)
        self.clock += 1
        (obs, reward, done) = (self.preprocess_obs(out[0]), out[1], float(out[2]))
        reset = done == 1.0 or self.clock == MAX_FRAMES
        return (obs, reward, done, reset)

    def render_rollout(self, states):
        if False:
            return 10
        self.internal_env.reset()
        for state in states:
            self.internal_env.env.state = state
            self.internal_env.render()

class CartPoleWrapper(GymWrapper):
    """
  Wrap CartPole.
  """

    def sample(self):
        if False:
            return 10
        return np.array([np.random.uniform(0.0, 1.0)])

    def normalize_actions(self, action):
        if False:
            print('Hello World!')
        return 1 if action[0] >= 0 else 0

    def unnormalize_actions(self, action):
        if False:
            print('Hello World!')
        return 2.0 * action - 1.0

class NoTimeLimitMujocoWrapper(GymWrapper):
    """
  Wrap Mujoco-style environments, removing the termination condition after time.
  This is needed to keep it Markovian.
  """

    def __init__(self, env_name):
        if False:
            while True:
                i = 10
        self.internal_env = gym.make(env_name).env
        self.observation_space = self.internal_env.observation_space
        self.action_space = self.internal_env.action_space
        self.custom_init()