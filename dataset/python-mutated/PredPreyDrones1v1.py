from copy import deepcopy
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym_pybullet_drones.utils.utils import sync
import pybullet as p
import os
from gym_pybullet_drones.utils.Logger import Logger
import time
ACTION2D = False

def draw_point(point, size=0.1, **kwargs):
    if False:
        while True:
            i = 10
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
        i = 10
        return i + 15
    assert len(start) == 3 and len(end) == 3
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width, parentObjectUniqueId=parent, parentLinkIndex=parent_link)

class Behavior:

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.kwargs = kwargs

    def fixed_prey(self, action, time, observation):
        if False:
            return 10
        if isinstance(action, dict):
            action[1] = [0, 0, 0]
        else:
            action[3:] = [0, 0, 0]
        return action

    def fixed_pred(self, action, time, observation):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(action, dict):
            action[0] = [0, 0, 0]
        else:
            action[:3] = [0, 0, 0]
        return action

    def cos_1D(self, action, time, observation):
        if False:
            print('Hello World!')
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
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        return self.compute_motion(0)

    def pos_x(self):
        if False:
            i = 10
            return i + 15
        return self.compute_motion(1)

    def neg_x(self):
        if False:
            i = 10
            return i + 15
        return self.compute_motion(2)

    def pos_y(self):
        if False:
            for i in range(10):
                print('nop')
        return self.compute_motion(3)

    def neg_y(self):
        if False:
            while True:
                i = 10
        return self.compute_motion(4)

    def pos_z(self):
        if False:
            while True:
                i = 10
        return self.compute_motion(5)

    def neg_z(self):
        if False:
            for i in range(10):
                print('nop')
        return self.compute_motion(6)

class PredPreyDrones(BaseMultiagentAviary):

    def __init__(self, caught_distance=0.13, max_num_steps=1000, crashing_max_angle=np.pi / 4, pred_behavior=None, prey_behavior=None, pred_policy=None, prey_policy=None, seed_val=45, reward_type='normal', drone_model: DroneModel=DroneModel.CF2X, num_pred_drones: int=1, num_prey_drones: int=1, neighbourhood_radius: float=np.inf, initial_xyzs=None, initial_rpys=None, physics: Physics=Physics.PYB, freq: int=240, aggregate_phy_steps: int=1, gui=False, record=False, obs: ObservationType=ObservationType.KIN, act: ActionType=ActionType.PID, logger=False):
        if False:
            return 10
        self.nrobots = num_pred_drones + num_prey_drones
        if initial_xyzs is None:
            initial_xyzs = np.vstack((np.zeros((num_pred_drones, 3)), np.zeros((num_prey_drones, 3))))
        initial_xyzs = np.array([[0, -0.5, 0.2], [0, 0.5, 0.2]])
        BaseMultiagentAviary.__init__(self, drone_model=drone_model, num_drones=self.nrobots, neighbourhood_radius=neighbourhood_radius, initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics, freq=freq, aggregate_phy_steps=aggregate_phy_steps, gui=gui, record=record, obs=obs, act=act)
        self.reward_type = reward_type if reward_type is not None else 'normal'
        self.seed(seed_val)
        if ACTION2D:
            self.noutputs = 2
            low = []
            high = []
            for i in range(self.nrobots):
                a = self.action_space[i]
                low.extend([-1 for i in range(2)])
                high.extend([1 for i in range(2)])
            self.action_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        else:
            self.noutputs = self.action_space[0].shape[0]
            low = []
            high = []
            for i in range(self.nrobots):
                a = self.action_space[i]
                low.extend([l for l in a.low])
                high.extend([h for h in a.high])
            self.action_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.observation_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        low = []
        high = []
        for i in range(self.nrobots):
            o = self.observation_space[i]
            l = self.masking_observations([l for l in o.low])
            h = self.masking_observations([h for h in o.high])
            low.extend(list(l))
            high.extend(list(h))
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.ninputs = len(self.observation_mask) - sum(self.observation_mask)
        self.max_num_steps = max_num_steps
        self.pred_behavior = pred_behavior
        self.prey_behavior = prey_behavior
        self.pred_policy = pred_policy
        self.prey_policy = prey_policy
        self._set_env_parameters()
        self.caught_distance = caught_distance
        self.crashing_max_angle = crashing_max_angle
        self.observation = None
        self.logger = Logger(logging_freq_hz=int(self.SIM_FREQ / self.AGGR_PHY_STEPS), num_drones=self.nrobots) if logger else None
        self.reach_goal = np.array([0.5, 0, 0.5])

    def reinit(self, max_num_steps=1000, prey_behavior=None, pred_behavior=None):
        if False:
            print('Hello World!')
        self.max_num_steps = max_num_steps
        self.prey_behavior = prey_behavior
        self.pred_behavior = pred_behavior

    def seed(self, seed=None):
        if False:
            while True:
                i = 10
        (self.np_random, seed) = seeding.np_random(seed)
        print(f'Seed: {seed}\treward_type:{self.reward_type}')
        return [seed]

    def _addObstacles(self):
        if False:
            for i in range(10):
                print('nop')
        p.loadURDF(os.path.dirname(os.path.abspath(__file__)) + '/box.urdf', [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.CLIENT)
        BaseMultiagentAviary._addObstacles(self)

    def _set_env_parameters(self):
        if False:
            i = 10
            return i + 15
        self.num_steps = 0
        self.caught = False
        self.steps_done = False
        self.start_time = time.time()
        self.crashed = [False for i in range(self.nrobots)]
        self._pred_reward = None
        self._prey_reward = None

    def _add_whiskers(self):
        if False:
            print('Hello World!')
        pass

    def reset(self):
        if False:
            return 10
        observation = BaseMultiagentAviary.reset(self)
        self._set_env_parameters()
        self.observation = self._process_observation(observation)
        self.log()
        if self.GUI:
            draw_point(self.reach_goal)
        return self.observation

    def _get_agent_observation(self, obs):
        if False:
            for i in range(10):
                print('nop')
        return obs

    def _get_opponent_observation(self, obs):
        if False:
            i = 10
            return i + 15
        return obs

    def _process_action(self, action, observation):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        action : ndarray or list\n            The input action for all drones with empty/dummy actions for non-trained drone\n        Returns\n        -------\n        dict[int, ndarray]\n            (NUM_DRONES, 3)-shaped array of ints containing to clipped delta target positions\n        '
        ac = deepcopy(action)
        if self.prey_behavior is not None:
            ac = self.prey_behavior(ac, self.num_steps, observation)
        if self.pred_behavior is not None:
            ac = self.pred_behavior(ac, self.num_steps, observation)
        if self.pred_policy is not None:
            ac[:self.noutputs] = self.pred_policy.compute_action(self._get_opponent_observation(observation))
        if self.prey_policy is not None:
            ac[self.noutputs:] = self.prey_policy.compute_action(self._get_opponent_observation(observation))
        ac = [a * 6 for a in ac]
        if not ACTION2D:
            action_dict = {i: np.array(ac[self.noutputs * i:self.noutputs * (i + 1)]) for i in range(self.nrobots)}
        else:
            pred_z = self.pos[0, 2]
            prey_z = self.pos[1, 2]
            ac.insert(2, 6 * (-pred_z + 0.2))
            ac.append(6 * (-prey_z + 0.2))
            action_dict = {i: np.array(ac[(self.noutputs + 1) * i:(self.noutputs + 1) * (i + 1)]) for i in range(self.nrobots)}
        return action_dict

    def masking_observations(self, observations):
        if False:
            return 10
        masked_observations = np.ma.masked_array(observations, mask=self.observation_mask)
        result_observations = masked_observations.compressed()
        return result_observations

    def _process_observation(self, observation):
        if False:
            return 10
        ob = []
        for i in range(self.nrobots):
            ob.extend(self.masking_observations(observation[i]))
        return np.array(ob)

    def _process_reward(self, obs, action):
        if False:
            print('Hello World!')
        norm_action_predator = np.tanh(np.linalg.norm(action[:self.noutputs])) / 3
        norm_action_prey = np.tanh(np.linalg.norm(action[self.noutputs:])) / 3
        (prey_reward, predator_reward) = (None, None)
        if self.reward_type == 'normal':
            prey_reward = 1
            predator_reward = -1
        elif self.reward_type == 'action_norm_pen':
            prey_reward = 1 - norm_action_prey
            predator_reward = -1 - norm_action_predator
        elif self.reward_type == 'relative_distance':
            dist = self._compute_relative_distance(obs) - self.caught_distance * 0.8
            prey_reward = np.tanh(dist)
            predator_reward = -np.tanh(dist)
        if self.caught:
            prey_reward = -10
            predator_reward = 10
        if self.steps_done:
            prey_reward = 10
            predator_reward = -10
        if self.reward_type == 'reach':
            pos0 = np.array(obs[0:3])
            pos1 = np.array(obs[self.ninputs:self.ninputs + 3])
            dist_pred = np.linalg.norm(pos0 - self.reach_goal)
            dist_prey = np.linalg.norm(pos1 - self.reach_goal)
            prey_reward = -dist_prey
            predator_reward = -dist_pred
            if dist_pred < 0.2:
                predator_reward = 10
            if dist_prey < 0.2:
                prey_reward = 10
        (self._pred_reward, self._prey_reward) = (predator_reward, prey_reward)
        return (predator_reward, prey_reward)

    def _compute_relative_distance(self, obs):
        if False:
            for i in range(10):
                print('nop')
        pos0 = np.array(obs[0:3])
        pos1 = np.array(obs[self.ninputs:self.ninputs + 3])
        dist = np.linalg.norm(pos0 - pos1)
        return dist

    def _compute_caught(self, obs):
        if False:
            return 10
        dist = self._compute_relative_distance(obs)
        if dist <= self.caught_distance:
            return True
        if self.reward_type == 'reach':
            pos0 = np.array(obs[0:3])
            pos1 = np.array(obs[self.ninputs:self.ninputs + 3])
            dist_pred = np.linalg.norm(pos0 - self.reach_goal)
            dist_prey = np.linalg.norm(pos1 - self.reach_goal)
            if dist_pred < 0.2 or dist_prey < 0.2:
                return True
        return False

    def _compute_crash(self, obs):
        if False:
            while True:
                i = 10
        crashed = [False for i in range(self.nrobots)]
        for i in range(self.nrobots):
            (roll, pitch, yaw) = self.rpy[i, :]
            if abs(roll) >= self.crashing_max_angle or abs(pitch) >= self.crashing_max_angle:
                crashed[i] = True
        return crashed

    def _process_done(self, obs):
        if False:
            i = 10
            return i + 15
        self.caught = self._compute_caught(obs)
        self.crashed = self._compute_crash(obs)
        self.steps_done = self.num_steps > self.max_num_steps
        done = True if self.caught or self.steps_done else False
        return done

    def who_won(self):
        if False:
            while True:
                i = 10
        if self.caught:
            return 'pred'
        if self.steps_done:
            return 'prey'
        if all(self.crashed):
            return 'none'
        elif self.crashed[0]:
            return 'prey'
        elif self.crashed[1]:
            return 'pred'
        return ''

    def _process_info(self):
        if False:
            for i in range(10):
                print('nop')
        return {'win': self.who_won(), 'crash': self.crashed, 'reward': (self._pred_reward, self._prey_reward)}

    def _computeReward(self):
        if False:
            return 10
        return 0

    def _computeDone(self):
        if False:
            i = 10
            return i + 15
        return False

    def _computeInfo(self):
        if False:
            print('Hello World!')
        return {}

    def step(self, action):
        if False:
            while True:
                i = 10
        self.num_steps += 1
        action_dict = self._process_action(action, self.observation)
        (obs_dict, _, _, _) = BaseMultiagentAviary.step(self, action_dict)
        obs = self._process_observation(obs_dict)
        self.observation = obs
        done = self._process_done(obs)
        reward = self._process_reward(obs, action)
        info = self._process_info()
        if done:
            print(info)
        self.log()
        return (obs, reward, done, info)

    def log(self):
        if False:
            i = 10
            return i + 15
        if self.logger is not None:
            for j in range(self.nrobots):
                state = self._getDroneStateVector(j)
                self.logger.log(drone=j, timestamp=self.num_steps / self.SIM_FREQ, state=state)

    def save_log(self):
        if False:
            for i in range(10):
                print('nop')
        self.logger.save()
        self.logger.save_as_csv('pid')

    def show_log(self):
        if False:
            for i in range(10):
                print('nop')
        self.logger.plot()

    def render(self, mode='human', extra_info=None):
        if False:
            i = 10
            return i + 15
        BaseMultiagentAviary.render(self, mode)
        sync(min(0, self.num_steps), self.start_time, self.TIMESTEP)

    def _clipAndNormalizeState(self, state):
        if False:
            return 10
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
            return 10
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

class PredPrey1v1PredDrone(PredPreyDrones, gym.Env):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        PredPreyDrones.__init__(self, **kwargs)
        self.action_space = spaces.Box(low=self.action_space.low[:self.noutputs], high=self.action_space.high[:self.noutputs], dtype=np.float32)

    def _process_action(self, action, observation):
        if False:
            return 10
        if self.prey_behavior is None and self.prey_policy is None:
            raise ValueError('prey_behavior or prey_policy should be specified')
        if ACTION2D:
            action = np.array([action, [0, 0]]).flatten()
        else:
            action = np.array([action, [0, 0, 0]]).flatten()
        return PredPreyDrones._process_action(self, action, observation)

    def _process_observation(self, observation):
        if False:
            while True:
                i = 10
        return PredPreyDrones._process_observation(self, observation)

    def _get_opponent_observation(self, observation):
        if False:
            i = 10
            return i + 15
        return observation

    def who_won(self):
        if False:
            return 10
        if self.caught:
            return 1
        if self.steps_done:
            return -1
        if all(self.crashed):
            return 0
        elif self.crashed[0]:
            return -1
        elif self.crashed[1]:
            return 1
        return 0

    def _process_reward(self, ob, action):
        if False:
            while True:
                i = 10
        (predator_reward, prey_reward) = PredPreyDrones._process_reward(self, ob, action)
        return predator_reward

class PredPrey1v1PreyDrone(PredPreyDrones, gym.Env):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        PredPreyDrones.__init__(self, **kwargs)
        self.action_space = spaces.Box(low=self.action_space.low[self.noutputs:], high=self.action_space.high[self.noutputs:], dtype=np.float32)

    def _process_action(self, action, observation):
        if False:
            while True:
                i = 10
        if self.pred_behavior is None and self.pred_policy is None:
            raise ValueError('prey_behavior or prey_policy should be specified')
        if ACTION2D:
            action = np.array([[0, 0], action]).flatten()
        else:
            action = np.array([[0, 0, 0], action]).flatten()
        return PredPreyDrones._process_action(self, action, observation)

    def _process_observation(self, observation):
        if False:
            while True:
                i = 10
        return PredPreyDrones._process_observation(self, observation)

    def _get_opponent_observation(self, observation):
        if False:
            return 10
        return observation

    def who_won(self):
        if False:
            print('Hello World!')
        if self.caught:
            return -1
        if self.steps_done:
            return 1
        if all(self.crashed):
            return 0
        elif self.crashed[0]:
            return 1
        elif self.crashed[1]:
            return -1
        return 0

    def _process_reward(self, ob, action):
        if False:
            i = 10
            return i + 15
        (predator_reward, prey_reward) = PredPreyDrones._process_reward(self, ob, action)
        return prey_reward
if __name__ == '__main__':
    import gym
    from time import sleep
    from math import cos, sin, tan
    env = PredPreyDrones(seed_val=45, gui=True, logger=False, reward_type='reach')
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        action = [0.5, 0, 0, 0, 0, 0]
        print(f'Actions: {action}')
        (observation, reward, done, info) = env.step(action)
        total_reward += reward[0]
        print(observation)
        print(reward, info, done)
    env.close()
    print(env.num_steps)
    print(total_reward)
    exit()
    env = PredPrey1v1PredDrone(seed_val=45, gui=True)
    behavior = Behavior()
    env.reinit(prey_behavior=behavior.fixed_prey)
    observation = env.reset()
    done = False
    while not done:
        action = [-observation[0] + observation[6], -observation[1] + observation[7], -observation[2] + observation[8]]
        if env.num_steps < 200:
            action = [0, 1, 0]
            print('Up')
        print(f'Actions: {action}')
        (observation, reward, done, info) = env.step(action)
        print(observation)
        print(reward, info, done)
        env.render()
    env.close()
    env = PredPrey1v1PreyDrone(seed_val=45, gui=True)
    behavior = Behavior()
    env.reinit(pred_behavior=behavior.fixed_pred)
    env.reset()
    done = False
    while not done:
        action = [0, 0, 1]
        if env.num_steps < 200:
            action = [0, 1, 0]
            print('Up')
        print(f'Actions: {action}')
        (observation, reward, done, info) = env.step(action)
        print(observation)
        print(reward, info, done)
        env.render()
    env.close()