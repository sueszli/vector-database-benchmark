import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig

class ParrotEnv(gym.Env):
    """Environment in which an agent must learn to repeat the seen observations.

    Observations are float numbers indicating the to-be-repeated values,
    e.g. -1.0, 5.1, or 3.2.

    The action space is always the same as the observation space.

    Rewards are r=-abs(observation - action), for all steps.
    """

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.action_space = config.get('parrot_shriek_range', gym.spaces.Box(-1.0, 1.0, shape=(1,)))
        self.observation_space = self.action_space
        self.cur_obs = None
        self.episode_len = 0

    def reset(self, *, seed=None, options=None):
        if False:
            for i in range(10):
                print('nop')
        'Resets the episode and returns the initial observation of the new one.'
        self.episode_len = 0
        self.cur_obs = self.observation_space.sample()
        return (self.cur_obs, {})

    def step(self, action):
        if False:
            for i in range(10):
                print('nop')
        'Takes a single step in the episode given `action`\n\n        Returns: New observation, reward, done-flag, info-dict (empty).\n        '
        self.episode_len += 1
        done = truncated = self.episode_len >= 10
        reward = -sum(abs(self.cur_obs - action))
        self.cur_obs = self.observation_space.sample()
        return (self.cur_obs, reward, done, truncated, {})
config = PPOConfig().environment(env=ParrotEnv, env_config={'parrot_shriek_range': gym.spaces.Box(-5.0, 5.0, (1,))}).rollouts(num_rollout_workers=3)
algo = config.build()
for i in range(5):
    results = algo.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
env = ParrotEnv({'parrot_shriek_range': gym.spaces.Box(-3.0, 3.0, (1,))})
(obs, info) = env.reset()
done = False
total_reward = 0.0
while not done:
    action = algo.compute_single_action(obs)
    (obs, reward, done, truncated, info) = env.step(action)
    total_reward += reward
print(f'Played 1 episode; total-reward={total_reward}')