import argparse
import gymnasium as gym
import numpy as np
import ray
from gymnasium.spaces import Box, Discrete
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import make_multi_agent
parser = argparse.ArgumentParser()
parser.add_argument('--framework', choices=['tf', 'tf2', 'torch'], default='torch', help='The DL framework specifier.')
parser.add_argument('--multi-agent', action='store_true')
parser.add_argument('--stop-iters', type=int, default=10)
parser.add_argument('--stop-timesteps', type=int, default=10000)
parser.add_argument('--stop-reward', type=float, default=9.0)

class CustomRenderedEnv(gym.Env):
    """Example of a custom env, for which you can specify rendering behavior."""
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.end_pos = config.get('corridor_length', 10)
        self.max_steps = config.get('max_steps', 100)
        self.cur_pos = 0
        self.steps = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 999.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if False:
            return 10
        self.cur_pos = 0.0
        self.steps = 0
        return ([self.cur_pos], {})

    def step(self, action):
        if False:
            for i in range(10):
                print('nop')
        self.steps += 1
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1.0
        elif action == 1:
            self.cur_pos += 1.0
        truncated = self.steps >= self.max_steps
        done = self.cur_pos >= self.end_pos or truncated
        return ([self.cur_pos], 10.0 if done else -0.1, done, truncated, {})

    def render(self, mode='rgb'):
        if False:
            i = 10
            return i + 15
        'Implements rendering logic for this env (given current state).\n\n        You can either return an RGB image:\n        np.array([height, width, 3], dtype=np.uint8) or take care of\n        rendering in a window yourself here (return True then).\n        For RLlib, though, only mode=rgb (returning an image) is needed,\n        even when "render_env" is True in the RLlib config.\n\n        Args:\n            mode: One of "rgb", "human", or "ascii". See gym.Env for\n                more information.\n\n        Returns:\n            Union[np.ndarray, bool]: An image to render or True (if rendering\n                is handled entirely in here).\n        '
        return np.random.randint(0, 256, size=(300, 400, 3), dtype=np.uint8)
MultiAgentCustomRenderedEnv = make_multi_agent(lambda config: CustomRenderedEnv(config))
if __name__ == '__main__':
    ray.init(num_cpus=4)
    args = parser.parse_args()
    config = PPOConfig().environment(MultiAgentCustomRenderedEnv if args.multi_agent else CustomRenderedEnv, env_config={'corridor_length': 10, 'max_steps': 100}).framework(args.framework).rollouts(num_envs_per_worker=2, num_rollout_workers=1).evaluation(evaluation_interval=1, evaluation_duration=2, evaluation_num_workers=1, evaluation_config=PPOConfig.overrides(render_env=True))
    stop = {'training_iteration': args.stop_iters, 'timesteps_total': args.stop_timesteps, 'episode_reward_mean': args.stop_reward}
    tune.Tuner('PPO', param_space=config.to_dict(), run_config=air.RunConfig(stop=stop)).fit()