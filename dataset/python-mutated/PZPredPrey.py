from bach_utils.logger import get_logger
clilog = get_logger()
from pettingzoo.mpe import simple_tag_v2
import numpy as np
from copy import deepcopy
import gym
from gym import spaces
from gym.utils import seeding
from torch import seed
import time
from math import copysign

class Behavior:

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.kwargs = kwargs

    def fixed_prey(self, action, time, observation):
        if False:
            i = 10
            return i + 15
        if isinstance(action, dict):
            action['agent_0'] = [0, 0, 0, 0, 0]
        else:
            action[2:] = [0, 0]
        return action

    def fixed_pred(self, action, time, observation):
        if False:
            i = 10
            return i + 15
        if isinstance(action, dict):
            action['adversary_0'] = [0, 0, 0, 0, 0]
        else:
            action[:2] = [0, 0]
        return action

class PZPredPrey(gym.Env):

    def __init__(self, max_num_steps=1000, pred_behavior=None, prey_behavior=None, pred_policy=None, prey_policy=None, seed_val=3, reward_type=None, caught_distance=0.001, gui=False, reseed=True, specific_pos={'adversary_0': np.array([-0.82870167, -0.52637899]), 'agent_0': np.array([0.60254893, 0]), 'landmark': [np.array([-0.73056844, -0.12037151]), np.array([-0.03770766, -0.61246995]), np.array([0.42223887, -0.69539036])]}):
        if False:
            return 10
        self.agent_keys = ['adversary_0', 'agent_0']
        self.nrobots = len(self.agent_keys)
        self.num_obstacles = 3
        self.specific_pos = specific_pos
        self.env = simple_tag_v2.parallel_env(num_good=1, num_adversaries=1, num_obstacles=self.num_obstacles, max_cycles=max_num_steps, continuous_actions=True, specific_pos=self.specific_pos)
        self.seed_val = seed_val
        self.reseed = reseed
        self.seed_val = self.seed(seed_val)[0]
        self.noutputs = 2
        low = []
        high = []
        for i in range(self.nrobots):
            low.extend([-1 for i in range(self.noutputs)])
            high.extend([1 for i in range(self.noutputs)])
        self.action_space_ = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.action_space = deepcopy(self.action_space_)
        low = []
        high = []
        rel_pos_limits = 4
        rel_vel_limits = 1
        self.normalized_obs_limits = [0.5, 0.5, 2, 2]
        self.normalized_obs_limits.extend([rel_pos_limits for _ in range(self.num_obstacles * 2)])
        self.normalized_obs_limits.extend([rel_pos_limits for _ in range(1 * 2)])
        self.normalized_obs_limits.extend([rel_vel_limits for _ in range(1 * 2)])
        self.normalized_obs_limits.append(1)
        self.ninputs = self.env.observation_space('adversary_0').shape[0] + 1
        for _ in range(self.nrobots):
            low.extend([-1 for i in range(self.ninputs)])
            high.extend([1 for i in range(self.ninputs)])
        self.observation_space_ = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.observation_space = deepcopy(self.observation_space_)
        self.caught_distance = caught_distance
        self.max_num_steps = max_num_steps
        self.pred_behavior = pred_behavior
        self.prey_behavior = prey_behavior
        self.pred_policy = pred_policy
        self.prey_policy = prey_policy
        self.reward_type = 'normal' if reward_type is None else reward_type
        self._set_env_parameters()
        self.caught = False
        self.steps_done = False
        self.observation = None
        self._posx_lim = [-1.8, 1.8]
        self._posy_lim = [-1.8, 1.8]

    def _set_env_parameters(self):
        if False:
            return 10
        self.num_steps = 0
        self.caught = False
        self.steps_done = False
        self._pred_reward = None
        self._prey_reward = None

    def reinit(self, max_num_steps=1000, pred_behavior=None, prey_behavior=None):
        if False:
            for i in range(10):
                print('nop')
        self.max_num_steps = max_num_steps
        self.prey_behavior = prey_behavior
        self.pred_behavior = pred_behavior

    def set_seed(self, seed_val):
        if False:
            print('Hello World!')
        if not self.reseed:
            self.seed_val = seed_val

    def seed(self, seed_val=None):
        if False:
            return 10
        (self.np_random, seed_val) = seeding.np_random(seed_val)
        clilog.debug(f'Seed (env): {self.seed_val}')
        clilog.warn(f'Warn: if you want to seed with different value, change seed_value of env first')
        print('Reset the env with seed function')
        self.env.reset(seed=self.seed_val, specific_pos=self.specific_pos)
        return [self.seed_val]

    def reset(self):
        if False:
            return 10
        obs = None
        if self.reseed:
            clilog.debug(f'Reseed env with the initial seed: {self.seed_val}')
            obs = self.env.reset(seed=self.seed_val, specific_pos=self.specific_pos)
        else:
            obs = self.env.reset(specific_pos=self.specific_pos)
        if self.specific_pos is not None:
            clilog.debug(f'Initialize the env with specific positions')
        self.num_steps = 0
        (self.observation, self.whole_observation) = self._process_observation(obs)
        return self.observation

    def _get_agent_observation(self, obs):
        if False:
            print('Hello World!')
        return obs

    def _get_opponent_observation(self, obs):
        if False:
            while True:
                i = 10
        raise NotImplementedError('_get_opponent_observation() Not implemented')

    def _transform_action(self, a):
        if False:
            return 10
        new_a = [0, 0, 0, 0, 0]
        idx_map = [(1, 2), (3, 4)]
        for i in range(2):
            idx = None
            if int(copysign(1, a[i])) > 0:
                idx = 0
            else:
                idx = 1
            new_a[idx_map[i][idx]] = abs(a[i])
        return new_a

    def _process_action(self, action, observation):
        if False:
            while True:
                i = 10
        '\n        Change the actions generated by the policy (List) to the base (PettingZoo) environment datatype (Dict)\n        ----------\n        Parameters\n        ----------\n        action : ndarray or list\n            Action from the policy\n        observation: list\n            Observations to be used by other agents to infer their actions as this environment is a single agent env while the others agents are preloaded polices\n        ----------\n        Returns\n        -------\n        dict[string, ndarray]\n        '
        ac = deepcopy(action)
        if self.prey_behavior is not None:
            ac = self.prey_behavior(ac, self.num_steps, observation)
        if self.pred_behavior is not None:
            ac = self.pred_behavior(ac, self.num_steps, observation)
        if self.pred_policy is not None:
            ac[:self.noutputs] = self.pred_policy.compute_action(self._get_opponent_observation(observation))
        if self.prey_policy is not None:
            ac[self.noutputs:] = self.prey_policy.compute_action(self._get_opponent_observation(observation))
        ac = [a for a in ac]
        action_dict = {self.agent_keys[i]: np.array(self._transform_action(ac[self.noutputs * i:self.noutputs * (i + 1)]), dtype=np.float32) for i in range(self.nrobots)}
        return action_dict

    def _normalize_obs(self, obs):
        if False:
            print('Hello World!')

        def normalize(o, mn, mx):
            if False:
                return 10
            return 2 * (o - mn) / (mx - mn) - 1
        normalized_obs = []
        for (i, o) in enumerate(obs):
            mn = -self.normalized_obs_limits[i]
            mx = -mn
            normalized_obs.append(normalize(o, mn, mx))
        return np.array(normalized_obs)

    def _process_observation(self, obs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Change from PZ environment's observations (dict) to list of observations\n        ----------\n        Parameters:\n        ----------\n            obs: dict[string, ndarray]\n        ----------\n        Returns:\n        ----------\n            obs_list: ndarray or list\n        ----------\n        "
        obs_list = []
        for i in range(self.nrobots):
            num_steps = self.num_steps / self.max_num_steps
            extended_obs = [num_steps]
            if i == self.nrobots - 1:
                extended_obs = [0 for _ in range(2)]
                extended_obs.append(num_steps)
            tmp_obs = np.append(obs[self.agent_keys[i]], extended_obs)
            normalized_obs = self._normalize_obs(tmp_obs)
            obs_list.extend(normalized_obs)
        ret_obs = np.array(obs_list, dtype=np.float32).flatten()
        return (ret_obs, ret_obs)

    def _process_reward(self, obs, action, reward_dict):
        if False:
            for i in range(10):
                print('nop')
        (prey_reward, predator_reward) = (reward_dict['agent_0'], reward_dict['adversary_0'])
        dist = 0
        timestep_reward = 3 * self.num_steps / self.max_num_steps
        prey_reward += 1 + timestep_reward + dist
        predator_reward += -1 - timestep_reward - dist
        if self.caught:
            prey_reward = -1000
            predator_reward = 1000
        if self.steps_done:
            prey_reward = 1000
            predator_reward = -1000
        (self._pred_reward, self._prey_reward) = (predator_reward, prey_reward)
        return (predator_reward, prey_reward)

    def _compute_caught(self, obs):
        if False:
            for i in range(10):
                print('nop')
        delta_pos = obs[self.num_obstacles * 2 + 4:self.num_obstacles * 2 + 6]
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = 0.04
        return True if dist < dist_min else False

    def _process_done(self, obs, done_dict, reward_dict):
        if False:
            print('Hello World!')
        self.caught = self._compute_caught(obs)
        self.steps_done = self.num_steps >= self.max_num_steps
        done = True if self.caught or self.steps_done else False
        return done

    def who_won(self):
        if False:
            print('Hello World!')
        if self.caught:
            return 'pred'
        if self.steps_done:
            return 'prey'
        return ''

    def _process_info(self, obs_dict):
        if False:
            while True:
                i = 10
        self.pred_pos = obs_dict[self.agent_keys[0]][2:4]
        self.prey_pos = obs_dict[self.agent_keys[1]][2:4]
        return {'win': self.who_won(), 'reward': (self._pred_reward, self._prey_reward), 'num_steps': self.num_steps, 'pred_pos': self.pred_pos, 'prey_pos': self.prey_pos}

    def step(self, action):
        if False:
            while True:
                i = 10
        self.num_steps += 1
        action_dict = self._process_action(action, self.whole_observation)
        (obs_dict, reward_dict, done_dict, info_dict) = self.env.step(action_dict)
        (obs, whole_obs) = self._process_observation(obs_dict)
        self.obs = obs
        self.whole_observation = whole_obs
        done = self._process_done(whole_obs, done_dict, reward_dict)
        reward = self._process_reward(obs, action, reward_dict)
        info = self._process_info(obs_dict)
        if done:
            clilog.debug(info)
        return (obs, reward, done, info)

    def render(self, mode='human', extra_info=None):
        if False:
            for i in range(10):
                print('nop')
        extra_info = f'{self.num_steps}' if extra_info is None else f'{self.num_steps}, ' + extra_info
        self.env.render(mode, extra_info)

    def close(self):
        if False:
            i = 10
            return i + 15
        time.sleep(0.3)
        self.env.close()

class PZPredPreyPred(PZPredPrey):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        PZPredPrey.__init__(self, **kwargs)
        self.action_space = spaces.Box(low=self.action_space_.low[:self.noutputs], high=self.action_space_.high[:self.noutputs], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.observation_space_.low[:self.ninputs], high=self.observation_space_.high[:self.ninputs], dtype=np.float32)

    def _process_action(self, action, observation):
        if False:
            for i in range(10):
                print('nop')
        if self.prey_behavior is None and self.prey_policy is None:
            raise ValueError('prey_behavior or prey_policy should be specified')
        action = np.array([action, [0 for _ in range(self.noutputs)]], dtype=np.float32).flatten()
        return PZPredPrey._process_action(self, action, observation)

    def _process_observation(self, observation):
        if False:
            print('Hello World!')
        (obs, _) = PZPredPrey._process_observation(self, observation)
        return (self._get_agent_observation(obs), obs)

    def _get_agent_observation(self, observation):
        if False:
            return 10
        return observation[:self.ninputs]

    def _get_opponent_observation(self, observation):
        if False:
            for i in range(10):
                print('nop')
        return observation[self.ninputs:]

    def who_won(self):
        if False:
            for i in range(10):
                print('nop')
        if self.caught:
            return 1
        if self.steps_done:
            return -1
        return 0

    def _process_reward(self, obs, action, reward_dict):
        if False:
            print('Hello World!')
        (predator_reward, prey_reward) = PZPredPrey._process_reward(self, obs, action, reward_dict)
        return predator_reward

class PZPredPreyPrey(PZPredPrey):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        PZPredPrey.__init__(self, **kwargs)
        self.action_space = spaces.Box(low=self.action_space_.low[self.noutputs:], high=self.action_space_.high[self.noutputs:], dtype=np.float32)
        self._ninputs = self.ninputs
        self.observation_space = spaces.Box(low=self.observation_space_.low[self.ninputs:self.ninputs + self._ninputs], high=self.observation_space_.high[self.ninputs:self.ninputs + self._ninputs], dtype=np.float32)

    def _process_action(self, action, observation):
        if False:
            return 10
        if self.pred_behavior is None and self.pred_policy is None:
            raise ValueError('pred_behavior or pred_policy should be specified')
        action = np.array([[0 for _ in range(self.noutputs)], action]).flatten()
        return PZPredPrey._process_action(self, action, observation)

    def _process_observation(self, observation):
        if False:
            for i in range(10):
                print('nop')
        (obs, _) = PZPredPrey._process_observation(self, observation)
        return (self._get_agent_observation(obs), obs)

    def _get_agent_observation(self, observation):
        if False:
            i = 10
            return i + 15
        return observation[self.ninputs:self.ninputs + self._ninputs]

    def _get_opponent_observation(self, observation):
        if False:
            i = 10
            return i + 15
        return observation[:self.ninputs]

    def who_won(self):
        if False:
            print('Hello World!')
        if self.caught:
            return -1
        if self.steps_done:
            return 1
        return 0

    def _process_reward(self, obs, action, reward_dict):
        if False:
            return 10
        (predator_reward, prey_reward) = PZPredPrey._process_reward(self, obs, action, reward_dict)
        return prey_reward

def print_obs(obs, n_landmarks):
    if False:
        print('Hello World!')
    print(f'Self vel: {obs[0:2]}')
    print(f'Self pos: {obs[2:4]}')
    print(f'Landmark rel pos: {obs[4:4 + n_landmarks * 2]}')
    print(4 + n_landmarks * 2, 4 + n_landmarks * 2 + 2)
    print(f'Other agents rel pos: {obs[4 + n_landmarks * 2:4 + n_landmarks * 2 + 2]}')
    print(f'Other agents rel vel: {obs[4 + n_landmarks * 2 + 2:4 + n_landmarks * 2 + 4]}')
    print(f'Time: {obs[4 + n_landmarks * 2 + 4:]}')
if __name__ == '__main__':
    import gym
    from time import sleep
    from matplotlib import pyplot as plt
    env = PZPredPreyPrey(seed_val=3)
    behavior = Behavior()
    env.reinit(pred_behavior=behavior.fixed_pred)
    for i in range(1):
        observation = env.reset()
        done = False
        rewards = []
        while not done:
            action = [-0.5, 0]
            (observation, reward, done, info) = env.step(action)
            print(env.prey_pos)
            rewards.append(info['reward'][1])
            env.render(extra_info='test')
            sleep(0.01)
        env.close()
    print(f'Sum: {sum(rewards)}')
    print(f'Max: {max(rewards)}')
    print(f'Min: {min(rewards)}')
    plt.plot(rewards)
    plt.show()