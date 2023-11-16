from typing import Any, List, Union, Optional, Dict
import gymnasium as gym
import numpy as np
import pettingzoo
from functools import reduce
from ding.envs import BaseEnv, BaseEnvTimestep, FrameStackWrapper
from ding.torch_utils import to_ndarray, to_list
from ding.envs.common.common_function import affine_transform
from ding.utils import ENV_REGISTRY, import_module
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.mpe.simple_spread.simple_spread import Scenario

@ENV_REGISTRY.register('petting_zoo')
class PettingZooEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        if False:
            while True:
                i = 10
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        self._env_family = self._cfg.env_family
        self._env_id = self._cfg.env_id
        self._num_agents = self._cfg.n_agent
        self._num_landmarks = self._cfg.n_landmark
        self._continuous_actions = self._cfg.get('continuous_actions', False)
        self._max_cycles = self._cfg.get('max_cycles', 25)
        self._act_scale = self._cfg.get('act_scale', False)
        self._agent_specific_global_state = self._cfg.get('agent_specific_global_state', False)
        if self._act_scale:
            assert self._continuous_actions, 'Only continuous action space env needs act_scale'

    def reset(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        if not self._init_flag:
            _env = make_env(simple_spread_raw_env)
            parallel_env = parallel_wrapper_fn(_env)
            self._env = parallel_env(N=self._cfg.n_agent, continuous_actions=self._continuous_actions, max_cycles=self._max_cycles)
        if self._replay_path is not None:
            self._env = gym.wrappers.Monitor(self._env, self._replay_path, video_callable=lambda episode_id: True, force=True)
        if hasattr(self, '_seed'):
            obs = self._env.reset(seed=self._seed)
        else:
            obs = self._env.reset()
        if not self._init_flag:
            self._agents = self._env.agents
            self._action_space = gym.spaces.Dict({agent: self._env.action_space(agent) for agent in self._agents})
            single_agent_obs_space = self._env.action_space(self._agents[0])
            if isinstance(single_agent_obs_space, gym.spaces.Box):
                self._action_dim = single_agent_obs_space.shape
            elif isinstance(single_agent_obs_space, gym.spaces.Discrete):
                self._action_dim = (single_agent_obs_space.n,)
            else:
                raise Exception('Only support `Box` or `Discrete` obs space for single agent.')
            if not self._cfg.agent_obs_only:
                self._observation_space = gym.spaces.Dict({'agent_state': gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self._num_agents, self._env.observation_space('agent_0').shape[0]), dtype=np.float32), 'global_state': gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(4 * self._num_agents + 2 * self._num_landmarks + 2 * self._num_agents * (self._num_agents - 1),), dtype=np.float32), 'agent_alone_state': gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self._num_agents, 4 + 2 * self._num_landmarks + 2 * (self._num_agents - 1)), dtype=np.float32), 'agent_alone_padding_state': gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self._num_agents, self._env.observation_space('agent_0').shape[0]), dtype=np.float32), 'action_mask': gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self._num_agents, self._action_dim[0]), dtype=np.float32)})
                if self._agent_specific_global_state:
                    agent_specifig_global_state = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self._num_agents, self._env.observation_space('agent_0').shape[0] + 4 * self._num_agents + 2 * self._num_landmarks + 2 * self._num_agents * (self._num_agents - 1)), dtype=np.float32)
                    self._observation_space['global_state'] = agent_specifig_global_state
            else:
                self._observation_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self._num_agents, self._env.observation_space('agent_0').shape[0]), dtype=np.float32)
            self._reward_space = gym.spaces.Dict({agent: gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,), dtype=np.float32) for agent in self._agents})
            self._init_flag = True
        self._eval_episode_return = 0.0
        self._step_count = 0
        obs_n = self._process_obs(obs)
        return obs_n

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def render(self) -> None:
        if False:
            i = 10
            return i + 15
        self._env.render()

    def seed(self, seed: int, dynamic_seed: bool=True) -> None:
        if False:
            print('Hello World!')
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        if False:
            i = 10
            return i + 15
        self._step_count += 1
        assert isinstance(action, np.ndarray), type(action)
        action = self._process_action(action)
        if self._act_scale:
            for agent in self._agents:
                action[agent] = affine_transform(action[agent], min_val=self.action_space[agent].low, max_val=self.action_space[agent].high)
        (obs, rew, done, trunc, info) = self._env.step(action)
        obs_n = self._process_obs(obs)
        rew_n = np.array([sum([rew[agent] for agent in self._agents])])
        rew_n = rew_n.astype(np.float32)
        self._eval_episode_return += rew_n.item()
        done_n = reduce(lambda x, y: x and y, done.values()) or self._step_count >= self._max_cycles
        if done_n:
            info['eval_episode_return'] = self._eval_episode_return
        return BaseEnvTimestep(obs_n, rew_n, done_n, info)

    def enable_save_replay(self, replay_path: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def _process_obs(self, obs: 'torch.Tensor') -> np.ndarray:
        if False:
            print('Hello World!')
        obs = np.array([obs[agent] for agent in self._agents]).astype(np.float32)
        if self._cfg.get('agent_obs_only', False):
            return obs
        ret = {}
        ret['agent_state'] = obs
        ret['global_state'] = np.concatenate([obs[0, 2:-(self._num_agents - 1) * 2], obs[:, 0:2].flatten(), obs[:, -(self._num_agents - 1) * 2:].flatten()])
        if self._agent_specific_global_state:
            ret['global_state'] = np.concatenate([ret['agent_state'], np.expand_dims(ret['global_state'], axis=0).repeat(self._num_agents, axis=0)], axis=1)
        ret['agent_alone_state'] = np.concatenate([obs[:, 0:4 + self._num_agents * 2], obs[:, -(self._num_agents - 1) * 2:]], 1)
        ret['agent_alone_padding_state'] = np.concatenate([obs[:, 0:4 + self._num_agents * 2], np.zeros((self._num_agents, (self._num_agents - 1) * 2), np.float32), obs[:, -(self._num_agents - 1) * 2:]], 1)
        ret['action_mask'] = np.ones((self._num_agents, *self._action_dim)).astype(np.float32)
        return ret

    def _process_action(self, action: 'torch.Tensor') -> Dict[str, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        dict_action = {}
        for (i, agent) in enumerate(self._agents):
            agent_action = action[i]
            if agent_action.shape == (1,):
                agent_action = agent_action.squeeze()
            dict_action[agent] = agent_action
        return dict_action

    def random_action(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        random_action = self.action_space.sample()
        for k in random_action:
            if isinstance(random_action[k], np.ndarray):
                pass
            elif isinstance(random_action[k], int):
                random_action[k] = to_ndarray([random_action[k]], dtype=np.int64)
        return random_action

    def __repr__(self) -> str:
        if False:
            return 10
        return 'DI-engine PettingZoo Env'

    @property
    def agents(self) -> List[str]:
        if False:
            return 10
        return self._agents

    @property
    def observation_space(self) -> gym.spaces.Space:
        if False:
            return 10
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        if False:
            while True:
                i = 10
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        if False:
            i = 10
            return i + 15
        return self._reward_space

class simple_spread_raw_env(SimpleEnv):

    def __init__(self, N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False):
        if False:
            for i in range(10):
                print('nop')
        assert 0.0 <= local_ratio <= 1.0, 'local_ratio is a proportion. Must be between 0 and 1.'
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions=continuous_actions, local_ratio=local_ratio)
        self.metadata['name'] = 'simple_spread_v2'

    def _execute_world_step(self):
        if False:
            return 10
        for (i, agent) in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])
        self.world.step()
        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))
        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = global_reward + agent_reward
            else:
                reward = agent_reward
            self.rewards[agent.name] = reward