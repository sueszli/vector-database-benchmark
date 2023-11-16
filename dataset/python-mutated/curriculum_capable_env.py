import gymnasium as gym
import random
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override

class CurriculumCapableEnv(TaskSettableEnv):
    """Example of a curriculum learning capable env.

    This simply wraps a FrozenLake-v1 env and makes it harder with each
    task. Task (difficulty levels) can range from 1 to 10."""
    MAPS = [['SFFFFFF', 'FFFFFFF', 'FFFFFFF', 'HHFFFFG', 'FFFFFFF', 'FFFFFFF'], ['SFFFFFF', 'FFFHFFF', 'FFFFFFF', 'HHHFFFF', 'FFFFFFG', 'FFFFFFF'], ['SFFFFFF', 'FFHHFFF', 'FFFFFFF', 'HHHHFFF', 'FFFFFFF', 'FFFFFFG'], ['SFFFFFF', 'FHHHFFF', 'FFFFFFF', 'HHHHHFF', 'FFFFFFF', 'FFFFFGF'], ['SFFFFFF', 'FFFHHFF', 'FHFFFFF', 'HHHHHHF', 'FFHFFHF', 'FFFGFFF']]

    def __init__(self, config: EnvContext):
        if False:
            for i in range(10):
                print('nop')
        self.cur_level = config.get('start_level', 1)
        self.max_timesteps = config.get('max_timesteps', 18)
        self.frozen_lake = None
        self._make_lake()
        self.observation_space = self.frozen_lake.observation_space
        self.action_space = self.frozen_lake.action_space
        self.switch_env = False
        self._timesteps = 0

    def reset(self, *, seed=None, options=None):
        if False:
            return 10
        if self.switch_env:
            self.switch_env = False
            self._make_lake()
        self._timesteps = 0
        return self.frozen_lake.reset(seed=seed, options=options)

    def step(self, action):
        if False:
            return 10
        self._timesteps += 1
        (obs, rew, done, truncated, info) = self.frozen_lake.step(action)
        rew *= 10 ** (self.cur_level - 1)
        if self._timesteps >= self.max_timesteps:
            done = True
        return (obs, rew, done, truncated, info)

    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        if False:
            for i in range(10):
                print('nop')
        'Implement this to sample n random tasks.'
        return [random.randint(1, 10) for _ in range(n_tasks)]

    @override(TaskSettableEnv)
    def get_task(self):
        if False:
            return 10
        'Implement this to get the current task (curriculum level).'
        return self.cur_level

    @override(TaskSettableEnv)
    def set_task(self, task):
        if False:
            return 10
        'Implement this to set the task (curriculum level) for this env.'
        self.cur_level = task
        self.switch_env = True

    def _make_lake(self):
        if False:
            return 10
        self.frozen_lake = gym.make('FrozenLake-v1', desc=self.MAPS[self.cur_level - 1], is_slippery=False)