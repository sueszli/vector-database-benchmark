from copy import deepcopy
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary, ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync
import pybullet as p
import os
from gym_pybullet_drones.utils.Logger import Logger
import time
ACTION2D = False

def draw_point(point, size=0.1, **kwargs):
    if False:
        return 10
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size / 2 * axis
        p2 = np.array(point) + size / 2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines

def add_line(start, end, color=[0, 0, 0], width=1, lifetime=None, parent=-1, parent_link=-1):
    if False:
        for i in range(10):
            print('nop')
    assert len(start) == 3 and len(end) == 3
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width, parentObjectUniqueId=parent, parentLinkIndex=parent_link)

class Behavior:

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.kwargs = kwargs

    def fixed_prey(self, action, time, observation):
        if False:
            i = 10
            return i + 15
        if isinstance(action, dict):
            action[1] = [0, 0, 0]
        else:
            action[3:] = [0, 0, 0]
        return action

    def fixed_pred(self, action, time, observation):
        if False:
            i = 10
            return i + 15
        if isinstance(action, dict):
            action[0] = [0, 0, 0]
        else:
            action[:3] = [0, 0, 0]
        return action

    def cos_1D(self, action, time, observation):
        if False:
            return 10
        freq = 5
        amplitude = 1
        if isinstance(action, dict):
            action[1] = [0, 0]
        else:
            sin_wave = amplitude * np.cos(time / freq)
            action[2:] = [sin_wave, sin_wave]
        return action

class MotionPrimitive:

    def __init__(self, n_steps=5, max_val=0.5):
        if False:
            return 10
        self.n_steps = n_steps
        self.max_val = max_val
        self.directions = {0: [0, 0, 0], 1: [+1, 0, 0], 2: [-1, 0, 0], 3: [0, +1, 0], 4: [0, -1, 0], 5: [0, 0, +1], 6: [0, 0, -1]}
        self.num_motion_primitives = len(self.directions.keys())

    def compute_motion(self, idx):
        if False:
            for i in range(10):
                print('nop')
        return np.array(self.directions[idx], dtype=np.float32) * self.max_val

    def stop(self):
        if False:
            return 10
        return self.compute_motion(0)

    def pos_x(self):
        if False:
            print('Hello World!')
        return self.compute_motion(1)

    def neg_x(self):
        if False:
            print('Hello World!')
        return self.compute_motion(2)

    def pos_y(self):
        if False:
            while True:
                i = 10
        return self.compute_motion(3)

    def neg_y(self):
        if False:
            i = 10
            return i + 15
        return self.compute_motion(4)

    def pos_z(self):
        if False:
            for i in range(10):
                print('nop')
        return self.compute_motion(5)

    def neg_z(self):
        if False:
            print('Hello World!')
        return self.compute_motion(6)

class _DroneReach(BaseSingleAgentAviary):

    def __init__(self, caught_distance=0.01, max_num_steps=1000, crashing_max_angle=np.pi / 4, seed_val=45, reward_type='reach', drone_model: DroneModel=DroneModel.CF2X, initial_xyzs=None, initial_rpys=None, physics: Physics=Physics.PYB, freq: int=240, aggregate_phy_steps: int=1, gui=False, record=False, obs: ObservationType=ObservationType.KIN, act: ActionType=ActionType.PID, logger=False):
        if False:
            return 10
        if initial_xyzs is None:
            initial_xyzs = np.vstack((np.zeros((1, 3)),))
        initial_xyzs = np.array([[0, -0.5, 0.2]])
        BaseSingleAgentAviary.__init__(self, drone_model=drone_model, initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics, freq=freq, aggregate_phy_steps=aggregate_phy_steps, gui=gui, record=record, obs=obs, act=act)
        self.reward_type = reward_type if reward_type is not None else 'normal'
        self.seed(seed_val)
        self.nrobots = 1
        if ACTION2D:
            self.noutputs = 2
            low = []
            high = []
            a = self.action_space
            low.extend([-1 for i in range(2)])
            high.extend([1 for i in range(2)])
            self.action_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        else:
            self.noutputs = self.action_space.shape[0]
            low = []
            high = []
            a = self.action_space
            low.extend([l for l in a.low])
            high.extend([h for h in a.high])
            self.action_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.observation_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        low = []
        high = []
        o = self.observation_space
        l = self.masking_observations([l for l in o.low])
        h = self.masking_observations([h for h in o.high])
        low.extend(list(l))
        high.extend(list(h))
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.ninputs = len(self.observation_mask) - sum(self.observation_mask)
        self.max_num_steps = max_num_steps
        self._set_env_parameters()
        self.caught_distance = caught_distance
        self.crashing_max_angle = crashing_max_angle
        self.observation = None
        self.logger = Logger(logging_freq_hz=int(self.SIM_FREQ / self.AGGR_PHY_STEPS), num_drones=self.nrobots) if logger else None
        self.reach_goal = np.array([0.5, 0, 0.5])

    def seed(self, seed=None):
        if False:
            return 10
        (self.np_random, seed) = seeding.np_random(seed)
        print(f'Seed: {seed}\treward_type:{self.reward_type}')
        return [seed]

    def _addObstacles(self):
        if False:
            return 10
        p.loadURDF(os.path.dirname(os.path.abspath(__file__)) + '/box.urdf', [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.CLIENT)
        BaseSingleAgentAviary._addObstacles(self)

    def _set_env_parameters(self):
        if False:
            return 10
        self.num_steps = 0
        self.caught = False
        self.steps_done = False
        self.start_time = time.time()
        self.crashed = False
        self._reward = None

    def _add_whiskers(self):
        if False:
            i = 10
            return i + 15
        pass

    def reset(self):
        if False:
            print('Hello World!')
        observation = BaseSingleAgentAviary.reset(self)
        self._set_env_parameters()
        self.observation = self._process_observation(observation)
        self.log()
        if self.GUI:
            draw_point(self.reach_goal)
        return self.observation

    def _process_action(self, action):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        action : ndarray or list\n            The input action for all drones with empty/dummy actions for non-trained drone\n        Returns\n        -------\n        dict[int, ndarray]\n            (NUM_DRONES, 3)-shaped array of ints containing to clipped delta target positions\n        '
        ac = deepcopy(action)
        ac = [a * 6 for a in ac]
        if ACTION2D:
            z_pos = self.pos[0, 2]
            ac.append(6 * (-z_pos + 0.2))
        ac = np.array(ac)
        return ac

    def masking_observations(self, observations):
        if False:
            print('Hello World!')
        masked_observations = np.ma.masked_array(observations, mask=self.observation_mask)
        result_observations = masked_observations.compressed()
        return result_observations

    def _process_observation(self, observation):
        if False:
            return 10
        ob = []
        ob.extend(self.masking_observations(observation))
        return np.array(ob)

    def _process_reward(self, obs, action):
        if False:
            i = 10
            return i + 15
        reward = None
        if self.reward_type == 'reach':
            dist = self._compute_relative_distance(obs)
            reward = -10 * dist ** 2
            if self.caught:
                reward = 300
        self._reward = reward
        return reward

    def _compute_relative_distance(self, obs):
        if False:
            return 10
        pos0 = self.pos[0, 0:3]
        dist = np.linalg.norm(pos0 - self.reach_goal)
        return dist

    def _compute_caught(self, obs):
        if False:
            for i in range(10):
                print('nop')
        if self.reward_type == 'reach':
            dist = self._compute_relative_distance(obs)
            if dist <= self.caught_distance:
                return True
            return False

    def _compute_crash(self):
        if False:
            i = 10
            return i + 15
        crashed = False
        (roll, pitch, yaw) = self.rpy[0, :]
        if abs(roll) >= self.crashing_max_angle or abs(pitch) >= self.crashing_max_angle:
            crashed = True
        return crashed

    def _process_done(self, obs):
        if False:
            return 10
        self.caught = self._compute_caught(obs)
        self.crashed = self._compute_crash()
        self.steps_done = self.num_steps > self.max_num_steps
        done = True if self.caught or self.steps_done or self.crashed else False
        return done

    def who_won(self):
        if False:
            i = 10
            return i + 15
        if self.caught:
            return 1
        if self.steps_done:
            return -1
        elif self.crashed:
            return 0
        return ''

    def _process_info(self):
        if False:
            i = 10
            return i + 15
        return {'win': self.who_won(), 'crash': self.crashed, 'reward': self._reward, 'caught': self.caught, 'dist': self._compute_relative_distance(self.observation)}

    def _computeReward(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def _computeDone(self):
        if False:
            i = 10
            return i + 15
        return False

    def _computeInfo(self):
        if False:
            while True:
                i = 10
        return {}

    def step(self, action):
        if False:
            while True:
                i = 10
        self.num_steps += 1
        action = self._process_action(action)
        (obs, _, _, _) = BaseSingleAgentAviary.step(self, action)
        self.observation = obs
        done = self._process_done(obs)
        reward = self._process_reward(obs, action)
        info = self._process_info()
        if done:
            print(f'Inside step function: {info}')
        self.log()
        return (obs, reward, done, info)

    def log(self):
        if False:
            return 10
        if self.logger is not None:
            for j in range(self.nrobots):
                state = self._getDroneStateVector(j)
                self.logger.log(drone=j, timestamp=self.num_steps / self.SIM_FREQ, state=state)

    def save_log(self):
        if False:
            return 10
        self.logger.save()
        self.logger.save_as_csv('pid')

    def show_log(self):
        if False:
            while True:
                i = 10
        self.logger.plot()

    def render(self, mode='human', extra_info=None):
        if False:
            for i in range(10):
                print('nop')
        BaseSingleAgentAviary.render(self, mode)
        sync(min(0, self.num_steps), self.start_time, self.TIMESTEP)

    def _clipAndNormalizeState(self, state):
        if False:
            for i in range(10):
                print('nop')
        "Normalizes a drone's state to the [-1,1] range.\n\n        Parameters\n        ----------\n        state : ndarray\n            (20,)-shaped array of floats containing the non-normalized state of a single drone.\n\n        Returns\n        -------\n        ndarray\n            (20,)-shaped array of floats containing the normalized state of a single drone.\n\n        "
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 2
        MAX_XY = 1.8
        MAX_Z = 1
        MAX_PITCH_ROLL = np.pi
        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        if self.GUI:
            self._clipAndNormalizeStateWarning(state, clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z)
        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        norm_and_clipped = np.hstack([normalized_pos_xy, normalized_pos_z, state[3:7], normalized_rp, normalized_y, normalized_vel_xy, normalized_vel_z, normalized_ang_vel, state[16:20]]).reshape(20)
        return norm_and_clipped

    def _clipAndNormalizeStateWarning(self, state, clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z):
        if False:
            while True:
                i = 10
        'Debugging printouts associated to `_clipAndNormalizeState`.\n\n        Print a warning if values in a state vector is out of the clipping range.\n        \n        '
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print('[WARNING] it', self.step_counter, 'in _clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]'.format(state[0], state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print('[WARNING] it', self.step_counter, 'in _clipAndNormalizeState(), clipped z position [{:.2f}]'.format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print('[WARNING] it', self.step_counter, 'in _clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]'.format(state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print('[WARNING] it', self.step_counter, 'in _clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]'.format(state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print('[WARNING] it', self.step_counter, 'in _clipAndNormalizeState(), clipped z velocity [{:.2f}]'.format(state[12]))

class DroneReach(_DroneReach):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(DroneReach, self).__init__(*args, **kwargs)
        self.target_opponent_policy_name = None

    def set_target_opponent_policy_name(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def set_sampled_opponents(*args, **kwargs):
        if False:
            return 10
        pass

    def set_opponents_indicies(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    import gym
    from time import sleep
    from math import cos, sin, tan
    env = DroneReach(seed_val=45, gui=True, logger=False)
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        action = [0.5 * (-env.pos[0][i] + env.reach_goal[i]) for i in range(3)]
        (observation, reward, done, info) = env.step(action)
        total_reward += reward
        print(info)
    env.close()
    print(env.num_steps)
    print(total_reward)