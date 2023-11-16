from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

class HalfCheetahRandDirecEnv(MujocoEnv, EzPickle, TaskSettableEnv):
    """HalfCheetah Environment with two diff tasks, moving forwards or backwards

    Direction is defined as a scalar: +1.0 (forwards) or -1.0 (backwards)
    """

    def __init__(self, goal_direction=None):
        if False:
            while True:
                i = 10
        self.goal_direction = goal_direction if goal_direction else 1.0
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        EzPickle.__init__(self, goal_direction)

    def sample_tasks(self, n_tasks):
        if False:
            while True:
                i = 10
        return np.random.choice((-1.0, 1.0), (n_tasks,))

    def set_task(self, task):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            task: task of the meta-learning environment\n        '
        self.goal_direction = task

    def get_task(self):
        if False:
            return 10
        '\n        Returns:\n            task: task of the meta-learning environment\n        '
        return self.goal_direction

    def step(self, action):
        if False:
            return 10
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.5 * 0.1 * np.square(action).sum()
        reward_run = self.goal_direction * (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return (ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl))

    def _get_obs(self):
        if False:
            print('Hello World!')
        return np.concatenate([self.sim.data.qpos.flat[1:], self.sim.data.qvel.flat])

    def reset_model(self):
        if False:
            while True:
                i = 10
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs

    def viewer_setup(self):
        if False:
            return 10
        self.viewer.cam.distance = self.model.stat.extent * 0.5