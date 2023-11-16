from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

class AntRandGoalEnv(EzPickle, MujocoEnv, TaskSettableEnv):
    """Ant Environment that randomizes goals as tasks

    Goals are randomly sampled 2D positions
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.set_task(self.sample_tasks(1)[0])
        MujocoEnv.__init__(self, 'ant.xml', 5)
        EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        if False:
            for i in range(10):
                print('nop')
        a = np.random.random(n_tasks) * 2 * np.pi
        r = 3 * np.random.random(n_tasks) ** 0.5
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        if False:
            while True:
                i = 10
        '\n        Args:\n            task: task of the meta-learning environment\n        '
        self.goal_pos = task

    def get_task(self):
        if False:
            print('Hello World!')
        '\n        Returns:\n            task: task of the meta-learning environment\n        '
        return self.goal_pos

    def step(self, a):
        if False:
            return 10
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com('torso')
        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))
        ctrl_cost = 0.1 * np.square(a).sum()
        contact_cost = 0.5 * 0.001 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        done = False
        ob = self._get_obs()
        return (ob, reward, done, dict(reward_forward=goal_reward, reward_ctrl=-ctrl_cost, reward_contact=-contact_cost, reward_survive=survive_reward))

    def _get_obs(self):
        if False:
            while True:
                i = 10
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat, np.clip(self.sim.data.cfrc_ext, -1, 1).flat])

    def reset_model(self):
        if False:
            i = 10
            return i + 15
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.viewer.cam.distance = self.model.stat.extent * 0.5