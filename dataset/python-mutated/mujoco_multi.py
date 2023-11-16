from functools import partial
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np
from .multiagentenv import MultiAgentEnv
from .obsk import get_joints_at_kdist, get_parts_and_edges, build_obs

class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        if False:
            while True:
                i = 10
        action = (action + 1) / 2
        action *= self.action_space.high - self.action_space.low
        action += self.action_space.low
        return action

    def action(self, action_):
        if False:
            i = 10
            return i + 15
        return self._action(action_)

    def _reverse_action(self, action):
        if False:
            for i in range(10):
                print('nop')
        action -= self.action_space.low
        action /= self.action_space.high - self.action_space.low
        action = action * 2 - 1
        return action

class MujocoMulti(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(batch_size, **kwargs)
        self.add_agent_id = kwargs['env_args']['add_agent_id']
        self.scenario = kwargs['env_args']['scenario']
        self.agent_conf = kwargs['env_args']['agent_conf']
        (self.agent_partitions, self.mujoco_edges, self.mujoco_globals) = get_parts_and_edges(self.scenario, self.agent_conf)
        self.n_agents = len(self.agent_partitions)
        self.n_actions = max([len(l) for l in self.agent_partitions])
        self.obs_add_global_pos = kwargs['env_args'].get('obs_add_global_pos', False)
        self.agent_obsk = kwargs['env_args'].get('agent_obsk', None)
        self.agent_obsk_agents = kwargs['env_args'].get('agent_obsk_agents', False)
        if self.agent_obsk is not None:
            self.k_categories_label = kwargs['env_args'].get('k_categories')
            if self.k_categories_label is None:
                if self.scenario in ['Ant-v2', 'manyagent_ant']:
                    self.k_categories_label = 'qpos,qvel,cfrc_ext|qpos'
                elif self.scenario in ['Humanoid-v2', 'HumanoidStandup-v2']:
                    self.k_categories_label = 'qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos'
                elif self.scenario in ['Reacher-v2']:
                    self.k_categories_label = 'qpos,qvel,fingertip_dist|qpos'
                elif self.scenario in ['coupled_half_cheetah']:
                    self.k_categories_label = 'qpos,qvel,ten_J,ten_length,ten_velocity|'
                else:
                    self.k_categories_label = 'qpos,qvel|qpos'
            k_split = self.k_categories_label.split('|')
            self.k_categories = [k_split[k if k < len(k_split) else -1].split(',') for k in range(self.agent_obsk + 1)]
            self.global_categories_label = kwargs['env_args'].get('global_categories')
            self.global_categories = self.global_categories_label.split(',') if self.global_categories_label is not None else []
        if self.agent_obsk is not None:
            self.k_dicts = [get_joints_at_kdist(agent_id, self.agent_partitions, self.mujoco_edges, k=self.agent_obsk, kagents=False) for agent_id in range(self.n_agents)]
        self.episode_limit = self.args.episode_limit
        self.env_version = kwargs['env_args'].get('env_version', 2)
        if self.env_version == 2:
            try:
                self.wrapped_env = NormalizedActions(gym.make(self.scenario))
            except gym.error.Error:
                if self.scenario in ['manyagent_ant']:
                    from .manyagent_ant import ManyAgentAntEnv as this_env
                elif self.scenario in ['manyagent_swimmer']:
                    from .manyagent_swimmer import ManyAgentSwimmerEnv as this_env
                elif self.scenario in ['coupled_half_cheetah']:
                    from .coupled_half_cheetah import CoupledHalfCheetah as this_env
                else:
                    raise NotImplementedError('Custom env not implemented!')
                self.wrapped_env = NormalizedActions(TimeLimit(this_env(**kwargs['env_args']), max_episode_steps=self.episode_limit))
        else:
            assert False, 'not implemented!'
        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.episode_limit
        if gym.version.VERSION > '0.22.0':
            self.env = self.timelimit_env.env.env.env.env
        else:
            self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        self.obs_size = self.get_obs_size()
        self.n = self.n_agents
        self.observation_space = [Box(low=np.array([-10] * self.n_agents), high=np.array([10] * self.n_agents)) for _ in range(self.n_agents)]
        acdims = [len(ap) for ap in self.agent_partitions]
        self.action_space = tuple([Box(self.env.action_space.low[sum(acdims[:a]):sum(acdims[:a + 1])], self.env.action_space.high[sum(acdims[:a]):sum(acdims[:a + 1])]) for a in range(self.n_agents)])

    def step(self, actions):
        if False:
            return 10
        flat_actions = np.concatenate([actions[i][:self.action_space[i].low.shape[0]] for i in range(self.n_agents)])
        (obs_n, reward_n, done_n, info_n) = self.wrapped_env.step(flat_actions)
        self.steps += 1
        info = {}
        info.update(info_n)
        if done_n:
            if self.steps < self.episode_limit:
                info['episode_limit'] = False
            else:
                info['episode_limit'] = True
        obs = {'agent_state': self.get_obs(), 'global_state': self.get_state()}
        return (obs, reward_n, done_n, info)

    def get_obs(self):
        if False:
            i = 10
            return i + 15
        ' Returns all agent observat3ions in a list '
        obs_n = []
        for a in range(self.n_agents):
            obs_n.append(self.get_obs_agent(a))
        return np.array(obs_n).astype(np.float32)

    def get_obs_agent(self, agent_id):
        if False:
            for i in range(10):
                print('nop')
        if self.agent_obsk is None:
            return self.env._get_obs()
        else:
            return build_obs(self.env, self.k_dicts[agent_id], self.k_categories, self.mujoco_globals, self.global_categories, vec_len=getattr(self, 'obs_size', None))

    def get_obs_size(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns the shape of the observation '
        if self.agent_obsk is None:
            return self.get_obs_agent(0).size
        else:
            return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])

    def get_state(self, team=None):
        if False:
            for i in range(10):
                print('nop')
        state_n = []
        if self.add_agent_id:
            state = self.env._get_obs()
            for a in range(self.n_agents):
                agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
                agent_id_feats[a] = 1.0
                state_i = np.concatenate([state, agent_id_feats])
                state_n.append(state_i)
        else:
            for a in range(self.n_agents):
                state_n.append(self.env._get_obs())
        return np.array(state_n).astype(np.float32)

    def get_state_size(self):
        if False:
            while True:
                i = 10
        ' Returns the shape of the state'
        return len(self.get_state())

    def get_avail_actions(self):
        if False:
            print('Hello World!')
        return np.ones(shape=(self.n_agents, self.n_actions))

    def get_avail_agent_actions(self, agent_id):
        if False:
            print('Hello World!')
        ' Returns the available actions for agent_id '
        return np.ones(shape=(self.n_actions,))

    def get_total_actions(self):
        if False:
            i = 10
            return i + 15
        ' Returns the total number of actions an agent could ever take '
        return self.n_actions

    def get_stats(self):
        if False:
            while True:
                i = 10
        return {}

    def get_agg_stats(self, stats):
        if False:
            while True:
                i = 10
        return {}

    def reset(self, **kwargs):
        if False:
            return 10
        ' Returns initial observations and states'
        self.steps = 0
        self.timelimit_env.reset()
        obs = {'agent_state': self.get_obs(), 'global_state': self.get_state()}
        return obs

    def render(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.env.render(**kwargs)

    def close(self):
        if False:
            print('Hello World!')
        pass

    def seed(self, args):
        if False:
            print('Hello World!')
        pass

    def get_env_info(self):
        if False:
            i = 10
            return i + 15
        env_info = {'state_shape': self.get_state_size(), 'obs_shape': self.get_obs_size(), 'n_actions': self.get_total_actions(), 'n_agents': self.n_agents, 'episode_limit': self.episode_limit, 'action_spaces': self.action_space, 'actions_dtype': np.float32, 'normalise_actions': False}
        return env_info