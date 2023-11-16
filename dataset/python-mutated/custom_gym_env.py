import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig

class SimpleCorridor(gym.Env):

    def __init__(self, config):
        if False:
            return 10
        self.end_pos = config['corridor_length']
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(self.end_pos)

    def reset(self, *, seed=None, options=None):
        if False:
            i = 10
            return i + 15
        self.cur_pos = 0
        return (self.cur_pos, {})

    def step(self, action):
        if False:
            return 10
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        if self.cur_pos >= self.end_pos:
            return (0, 1.0, True, True, {})
        else:
            return (self.cur_pos, -0.1, False, False, {})
ray.init()
config = PPOConfig().environment(SimpleCorridor, env_config={'corridor_length': 5})
algo = config.build()
for _ in range(3):
    print(algo.train())
algo.stop()