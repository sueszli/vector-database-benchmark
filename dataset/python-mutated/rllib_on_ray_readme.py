import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig

class SimpleCorridor(gym.Env):
    """Corridor in which an agent must learn to move right to reach the exit.

    ---------------------
    | S | 1 | 2 | 3 | G |   S=start; G=goal; corridor_length=5
    ---------------------

    Possible actions to chose from are: 0=left; 1=right
    Observations are floats indicating the current field index, e.g. 0.0 for
    starting position, 1.0 for the field next to the starting position, etc..
    Rewards are -0.1 for all steps, except when reaching the goal (+1.0).
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.end_pos = config['corridor_length']
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, self.end_pos, shape=(1,))

    def reset(self, *, seed=None, options=None):
        if False:
            while True:
                i = 10
        'Resets the episode.\n\n        Returns:\n           Initial observation of the new episode and an info dict.\n        '
        self.cur_pos = 0
        return ([self.cur_pos], {})

    def step(self, action):
        if False:
            for i in range(10):
                print('nop')
        'Takes a single step in the episode given `action`.\n\n        Returns:\n            New observation, reward, terminated-flag, truncated-flag, info-dict (empty).\n        '
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        terminated = self.cur_pos >= self.end_pos
        truncated = False
        reward = 1.0 if terminated else -0.1
        return ([self.cur_pos], reward, terminated, truncated, {})
config = PPOConfig().environment(env=SimpleCorridor, env_config={'corridor_length': 28}).rollouts(num_rollout_workers=3)
algo = config.build()
for i in range(5):
    results = algo.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
env = SimpleCorridor({'corridor_length': 10})
(obs, info) = env.reset()
terminated = truncated = False
total_reward = 0.0
while not terminated and (not truncated):
    action = algo.compute_single_action(obs)
    (obs, reward, terminated, truncated, info) = env.step(action)
    total_reward += reward
print(f'Played 1 episode; total-reward={total_reward}')