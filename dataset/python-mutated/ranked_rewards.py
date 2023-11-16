from copy import deepcopy
import numpy as np

class RankedRewardsBuffer:

    def __init__(self, buffer_max_length, percentile):
        if False:
            for i in range(10):
                print('nop')
        self.buffer_max_length = buffer_max_length
        self.percentile = percentile
        self.buffer = []

    def add_reward(self, reward):
        if False:
            while True:
                i = 10
        if len(self.buffer) < self.buffer_max_length:
            self.buffer.append(reward)
        else:
            self.buffer = self.buffer[1:] + [reward]

    def normalize(self, reward):
        if False:
            while True:
                i = 10
        reward_threshold = np.percentile(self.buffer, self.percentile)
        if reward < reward_threshold:
            return -1.0
        else:
            return 1.0

    def get_state(self):
        if False:
            return 10
        return np.array(self.buffer)

    def set_state(self, state):
        if False:
            i = 10
            return i + 15
        if state is not None:
            self.buffer = list(state)

def get_r2_env_wrapper(env_creator, r2_config):
    if False:
        while True:
            i = 10

    class RankedRewardsEnvWrapper:

        def __init__(self, env_config):
            if False:
                i = 10
                return i + 15
            self.env = env_creator(env_config)
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            max_buffer_length = r2_config['buffer_max_length']
            percentile = r2_config['percentile']
            self.r2_buffer = RankedRewardsBuffer(max_buffer_length, percentile)
            if r2_config['initialize_buffer']:
                self._initialize_buffer(r2_config['num_init_rewards'])

        def _initialize_buffer(self, num_init_rewards=100):
            if False:
                print('Hello World!')
            for _ in range(num_init_rewards):
                (obs, info) = self.env.reset()
                terminated = truncated = False
                while not terminated and (not truncated):
                    mask = obs['action_mask']
                    probs = mask / mask.sum()
                    action = np.random.choice(np.arange(mask.shape[0]), p=probs)
                    (obs, reward, terminated, truncated, _) = self.env.step(action)
                self.r2_buffer.add_reward(reward)

        def step(self, action):
            if False:
                return 10
            (obs, reward, terminated, truncated, info) = self.env.step(action)
            if terminated or truncated:
                reward = self.r2_buffer.normalize(reward)
            return (obs, reward, terminated, truncated, info)

        def get_state(self):
            if False:
                while True:
                    i = 10
            state = {'env_state': self.env.get_state(), 'buffer_state': self.r2_buffer.get_state()}
            return deepcopy(state)

        def reset(self, *, seed=None, options=None):
            if False:
                return 10
            return self.env.reset()

        def set_state(self, state):
            if False:
                print('Hello World!')
            obs = self.env.set_state(state['env_state'])
            self.r2_buffer.set_state(state['buffer_state'])
            return obs
    return RankedRewardsEnvWrapper