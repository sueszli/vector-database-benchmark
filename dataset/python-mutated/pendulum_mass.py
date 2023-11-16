from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

class PendulumMassEnv(PendulumEnv, EzPickle, TaskSettableEnv):
    """PendulumMassEnv varies the weight of the pendulum

    Tasks are defined to be weight uniformly sampled between [0.5,2]
    """

    def sample_tasks(self, n_tasks):
        if False:
            i = 10
            return i + 15
        return np.random.uniform(low=0.5, high=2.0, size=(n_tasks,))

    def set_task(self, task):
        if False:
            return 10
        '\n        Args:\n            task: Task of the meta-learning environment (here: mass of\n                the pendulum).\n        '
        self.m = task

    def get_task(self):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            float: The current mass of the pendulum (self.m in the PendulumEnv\n                object).\n        '
        return self.m