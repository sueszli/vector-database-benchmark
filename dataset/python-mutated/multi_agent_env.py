import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import ExperimentalAPI, override, PublicAPI, DeveloperAPI
from ray.rllib.utils.typing import AgentID, EnvCreator, EnvID, EnvType, MultiAgentDict, MultiEnvDict
from ray.util import log_once
ENV_STATE = 'state'
logger = logging.getLogger(__name__)

@PublicAPI
class MultiAgentEnv(gym.Env):
    """An environment that hosts multiple independent agents.

    Agents are identified by (string) agent ids. Note that these "agents" here
    are not to be confused with RLlib Algorithms, which are also sometimes
    referred to as "agents" or "RL agents".

    The preferred format for action- and observation space is a mapping from agent
    ids to their individual spaces. If that is not provided, the respective methods'
    observation_space_contains(), action_space_contains(),
    action_space_sample() and observation_space_sample() have to be overwritten.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'observation_space'):
            self.observation_space = None
        if not hasattr(self, 'action_space'):
            self.action_space = None
        if not hasattr(self, '_agent_ids'):
            self._agent_ids = set()
        if not hasattr(self, '_action_space_in_preferred_format'):
            self._action_space_in_preferred_format = None
        if not hasattr(self, '_obs_space_in_preferred_format'):
            self._obs_space_in_preferred_format = None

    @PublicAPI
    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        if False:
            while True:
                i = 10
        'Resets the env and returns observations from ready agents.\n\n        Args:\n            seed: An optional seed to use for the new episode.\n\n        Returns:\n            New observations for each ready agent.\n\n        .. testcode::\n            :skipif: True\n\n            from ray.rllib.env.multi_agent_env import MultiAgentEnv\n            class MyMultiAgentEnv(MultiAgentEnv):\n                # Define your env here.\n            env = MyMultiAgentEnv()\n            obs, infos = env.reset(seed=42, options={})\n            print(obs)\n\n        .. testoutput::\n\n            {\n                "car_0": [2.4, 1.6],\n                "car_1": [3.4, -3.2],\n                "traffic_light_1": [0, 3, 5, 1],\n            }\n        '
        super().reset(seed=seed, options=options)

    @PublicAPI
    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        if False:
            i = 10
            return i + 15
        'Returns observations from ready agents.\n\n        The returns are dicts mapping from agent_id strings to values. The\n        number of agents in the env can vary over time.\n\n        Returns:\n            Tuple containing 1) new observations for\n            each ready agent, 2) reward values for each ready agent. If\n            the episode is just started, the value will be None.\n            3) Terminated values for each ready agent. The special key\n            "__all__" (required) is used to indicate env termination.\n            4) Truncated values for each ready agent.\n            5) Info values for each agent id (may be empty dicts).\n\n        .. testcode::\n            :skipif: True\n\n            env = ...\n            obs, rewards, terminateds, truncateds, infos = env.step(action_dict={\n                "car_0": 1, "car_1": 0, "traffic_light_1": 2,\n            })\n            print(rewards)\n\n            print(terminateds)\n\n            print(infos)\n\n        .. testoutput::\n\n            {\n                "car_0": 3,\n                "car_1": -1,\n                "traffic_light_1": 0,\n            }\n            {\n                "car_0": False,    # car_0 is still running\n                "car_1": True,     # car_1 is terminated\n                "__all__": False,  # the env is not terminated\n            }\n            {\n                "car_0": {},  # info for car_0\n                "car_1": {},  # info for car_1\n            }\n\n        '
        raise NotImplementedError

    @ExperimentalAPI
    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if the observation space contains the given key.\n\n        Args:\n            x: Observations to check.\n\n        Returns:\n            True if the observation space contains the given all observations\n                in x.\n        '
        if not hasattr(self, '_obs_space_in_preferred_format') or self._obs_space_in_preferred_format is None:
            self._obs_space_in_preferred_format = self._check_if_obs_space_maps_agent_id_to_sub_space()
        if self._obs_space_in_preferred_format:
            for (key, agent_obs) in x.items():
                if not self.observation_space[key].contains(agent_obs):
                    return False
            if not all((k in self.observation_space.spaces for k in x)):
                if log_once('possibly_bad_multi_agent_dict_missing_agent_observations'):
                    logger.warning('You environment returns observations that are MultiAgentDicts with incomplete information. Meaning that they only contain information on a subset of participating agents. Ignore this warning if this is intended, for example if your environment is a turn-based simulation.')
            return True
        logger.warning('observation_space_contains() of {} has not been implemented. You can either implement it yourself or bring the observation space into the preferred format of a mapping from agent ids to their individual observation spaces. '.format(self))
        return True

    @ExperimentalAPI
    def action_space_contains(self, x: MultiAgentDict) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if the action space contains the given action.\n\n        Args:\n            x: Actions to check.\n\n        Returns:\n            True if the action space contains all actions in x.\n        '
        if not hasattr(self, '_action_space_in_preferred_format') or self._action_space_in_preferred_format is None:
            self._action_space_in_preferred_format = self._check_if_action_space_maps_agent_id_to_sub_space()
        if self._action_space_in_preferred_format:
            return all((self.action_space[agent].contains(x[agent]) for agent in x))
        if log_once('action_space_contains'):
            logger.warning('action_space_contains() of {} has not been implemented. You can either implement it yourself or bring the observation space into the preferred format of a mapping from agent ids to their individual observation spaces. '.format(self))
        return True

    @ExperimentalAPI
    def action_space_sample(self, agent_ids: list=None) -> MultiAgentDict:
        if False:
            return 10
        'Returns a random action for each environment, and potentially each\n            agent in that environment.\n\n        Args:\n            agent_ids: List of agent ids to sample actions for. If None or\n                empty list, sample actions for all agents in the\n                environment.\n\n        Returns:\n            A random action for each environment.\n        '
        if not hasattr(self, '_action_space_in_preferred_format') or self._action_space_in_preferred_format is None:
            self._action_space_in_preferred_format = self._check_if_action_space_maps_agent_id_to_sub_space()
        if self._action_space_in_preferred_format:
            if agent_ids is None:
                agent_ids = self.get_agent_ids()
            samples = self.action_space.sample()
            return {agent_id: samples[agent_id] for agent_id in agent_ids if agent_id != '__all__'}
        logger.warning(f'action_space_sample() of {self} has not been implemented. You can either implement it yourself or bring the observation space into the preferred format of a mapping from agent ids to their individual observation spaces.')
        return {}

    @ExperimentalAPI
    def observation_space_sample(self, agent_ids: list=None) -> MultiEnvDict:
        if False:
            print('Hello World!')
        'Returns a random observation from the observation space for each\n        agent if agent_ids is None, otherwise returns a random observation for\n        the agents in agent_ids.\n\n        Args:\n            agent_ids: List of agent ids to sample actions for. If None or\n                empty list, sample actions for all agents in the\n                environment.\n\n        Returns:\n            A random action for each environment.\n        '
        if not hasattr(self, '_obs_space_in_preferred_format') or self._obs_space_in_preferred_format is None:
            self._obs_space_in_preferred_format = self._check_if_obs_space_maps_agent_id_to_sub_space()
        if self._obs_space_in_preferred_format:
            if agent_ids is None:
                agent_ids = self.get_agent_ids()
            samples = self.observation_space.sample()
            samples = {agent_id: samples[agent_id] for agent_id in agent_ids}
            return samples
        if log_once('observation_space_sample'):
            logger.warning('observation_space_sample() of {} has not been implemented. You can either implement it yourself or bring the observation space into the preferred format of a mapping from agent ids to their individual observation spaces. '.format(self))
        return {}

    @PublicAPI
    def get_agent_ids(self) -> Set[AgentID]:
        if False:
            print('Hello World!')
        'Returns a set of agent ids in the environment.\n\n        Returns:\n            Set of agent ids.\n        '
        if not isinstance(self._agent_ids, set):
            self._agent_ids = set(self._agent_ids)
        return self._agent_ids

    @PublicAPI
    def render(self) -> None:
        if False:
            print('Hello World!')
        'Tries to render the environment.'
        pass

    def with_agent_groups(self, groups: Dict[str, List[AgentID]], obs_space: gym.Space=None, act_space: gym.Space=None) -> 'MultiAgentEnv':
        if False:
            print('Hello World!')
        'Convenience method for grouping together agents in this env.\n\n        An agent group is a list of agent IDs that are mapped to a single\n        logical agent. All agents of the group must act at the same time in the\n        environment. The grouped agent exposes Tuple action and observation\n        spaces that are the concatenated action and obs spaces of the\n        individual agents.\n\n        The rewards of all the agents in a group are summed. The individual\n        agent rewards are available under the "individual_rewards" key of the\n        group info return.\n\n        Agent grouping is required to leverage algorithms such as Q-Mix.\n\n        Args:\n            groups: Mapping from group id to a list of the agent ids\n                of group members. If an agent id is not present in any group\n                value, it will be left ungrouped. The group id becomes a new agent ID\n                in the final environment.\n            obs_space: Optional observation space for the grouped\n                env. Must be a tuple space. If not provided, will infer this to be a\n                Tuple of n individual agents spaces (n=num agents in a group).\n            act_space: Optional action space for the grouped env.\n                Must be a tuple space. If not provided, will infer this to be a Tuple\n                of n individual agents spaces (n=num agents in a group).\n\n        .. testcode::\n            :skipif: True\n\n            from ray.rllib.env.multi_agent_env import MultiAgentEnv\n            class MyMultiAgentEnv(MultiAgentEnv):\n                # define your env here\n                ...\n            env = MyMultiAgentEnv(...)\n            grouped_env = env.with_agent_groups(env, {\n              "group1": ["agent1", "agent2", "agent3"],\n              "group2": ["agent4", "agent5"],\n            })\n\n        '
        from ray.rllib.env.wrappers.group_agents_wrapper import GroupAgentsWrapper
        return GroupAgentsWrapper(self, groups, obs_space, act_space)

    @PublicAPI
    def to_base_env(self, make_env: Optional[Callable[[int], EnvType]]=None, num_envs: int=1, remote_envs: bool=False, remote_env_batch_wait_ms: int=0, restart_failed_sub_environments: bool=False) -> 'BaseEnv':
        if False:
            i = 10
            return i + 15
        'Converts an RLlib MultiAgentEnv into a BaseEnv object.\n\n        The resulting BaseEnv is always vectorized (contains n\n        sub-environments) to support batched forward passes, where n may\n        also be 1. BaseEnv also supports async execution via the `poll` and\n        `send_actions` methods and thus supports external simulators.\n\n        Args:\n            make_env: A callable taking an int as input (which indicates\n                the number of individual sub-environments within the final\n                vectorized BaseEnv) and returning one individual\n                sub-environment.\n            num_envs: The number of sub-environments to create in the\n                resulting (vectorized) BaseEnv. The already existing `env`\n                will be one of the `num_envs`.\n            remote_envs: Whether each sub-env should be a @ray.remote\n                actor. You can set this behavior in your config via the\n                `remote_worker_envs=True` option.\n            remote_env_batch_wait_ms: The wait time (in ms) to poll remote\n                sub-environments for, if applicable. Only used if\n                `remote_envs` is True.\n            restart_failed_sub_environments: If True and any sub-environment (within\n                a vectorized env) throws any error during env stepping, we will try to\n                restart the faulty sub-environment. This is done\n                without disturbing the other (still intact) sub-environments.\n\n        Returns:\n            The resulting BaseEnv object.\n        '
        from ray.rllib.env.remote_base_env import RemoteBaseEnv
        if remote_envs:
            env = RemoteBaseEnv(make_env, num_envs, multiagent=True, remote_env_batch_wait_ms=remote_env_batch_wait_ms, restart_failed_sub_environments=restart_failed_sub_environments)
        else:
            env = MultiAgentEnvWrapper(make_env=make_env, existing_envs=[self], num_envs=num_envs, restart_failed_sub_environments=restart_failed_sub_environments)
        return env

    @DeveloperAPI
    def _check_if_obs_space_maps_agent_id_to_sub_space(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if obs space maps from agent ids to spaces of individual agents.'
        return hasattr(self, 'observation_space') and isinstance(self.observation_space, gym.spaces.Dict) and (set(self.observation_space.spaces.keys()) == self.get_agent_ids())

    @DeveloperAPI
    def _check_if_action_space_maps_agent_id_to_sub_space(self) -> bool:
        if False:
            while True:
                i = 10
        'Checks if action space maps from agent ids to spaces of individual agents.'
        return hasattr(self, 'action_space') and isinstance(self.action_space, gym.spaces.Dict) and (set(self.action_space.keys()) == self.get_agent_ids())

@PublicAPI
def make_multi_agent(env_name_or_creator: Union[str, EnvCreator]) -> Type['MultiAgentEnv']:
    if False:
        print('Hello World!')
    'Convenience wrapper for any single-agent env to be converted into MA.\n\n    Allows you to convert a simple (single-agent) `gym.Env` class\n    into a `MultiAgentEnv` class. This function simply stacks n instances\n    of the given ```gym.Env``` class into one unified ``MultiAgentEnv`` class\n    and returns this class, thus pretending the agents act together in the\n    same environment, whereas - under the hood - they live separately from\n    each other in n parallel single-agent envs.\n\n    Agent IDs in the resulting and are int numbers starting from 0\n    (first agent).\n\n    Args:\n        env_name_or_creator: String specifier or env_maker function taking\n            an EnvContext object as only arg and returning a gym.Env.\n\n    Returns:\n        New MultiAgentEnv class to be used as env.\n        The constructor takes a config dict with `num_agents` key\n        (default=1). The rest of the config dict will be passed on to the\n        underlying single-agent env\'s constructor.\n\n    .. testcode::\n        :skipif: True\n\n        from ray.rllib.env.multi_agent_env import make_multi_agent\n        # By gym string:\n        ma_cartpole_cls = make_multi_agent("CartPole-v1")\n        # Create a 2 agent multi-agent cartpole.\n        ma_cartpole = ma_cartpole_cls({"num_agents": 2})\n        obs = ma_cartpole.reset()\n        print(obs)\n\n        # By env-maker callable:\n        from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole\n        ma_stateless_cartpole_cls = make_multi_agent(\n           lambda config: StatelessCartPole(config))\n        # Create a 3 agent multi-agent stateless cartpole.\n        ma_stateless_cartpole = ma_stateless_cartpole_cls(\n           {"num_agents": 3})\n        print(obs)\n\n    .. testoutput::\n\n        {0: [...], 1: [...]}\n        {0: [...], 1: [...], 2: [...]}\n    '

    class MultiEnv(MultiAgentEnv):

        def __init__(self, config: EnvContext=None):
            if False:
                for i in range(10):
                    print('nop')
            MultiAgentEnv.__init__(self)
            if config is None:
                config = {}
            num = config.pop('num_agents', 1)
            if isinstance(env_name_or_creator, str):
                self.envs = [gym.make(env_name_or_creator) for _ in range(num)]
            else:
                self.envs = [env_name_or_creator(config) for _ in range(num)]
            self.terminateds = set()
            self.truncateds = set()
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space
            self._agent_ids = set(range(num))

        @override(MultiAgentEnv)
        def observation_space_sample(self, agent_ids: list=None) -> MultiAgentDict:
            if False:
                print('Hello World!')
            if agent_ids is None:
                agent_ids = list(range(len(self.envs)))
            obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}
            return obs

        @override(MultiAgentEnv)
        def action_space_sample(self, agent_ids: list=None) -> MultiAgentDict:
            if False:
                print('Hello World!')
            if agent_ids is None:
                agent_ids = list(range(len(self.envs)))
            actions = {agent_id: self.action_space.sample() for agent_id in agent_ids}
            return actions

        @override(MultiAgentEnv)
        def action_space_contains(self, x: MultiAgentDict) -> bool:
            if False:
                return 10
            if not isinstance(x, dict):
                return False
            return all((self.action_space.contains(val) for val in x.values()))

        @override(MultiAgentEnv)
        def observation_space_contains(self, x: MultiAgentDict) -> bool:
            if False:
                i = 10
                return i + 15
            if not isinstance(x, dict):
                return False
            return all((self.observation_space.contains(val) for val in x.values()))

        @override(MultiAgentEnv)
        def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
            if False:
                print('Hello World!')
            self.terminateds = set()
            self.truncateds = set()
            (obs, infos) = ({}, {})
            for (i, env) in enumerate(self.envs):
                (obs[i], infos[i]) = env.reset(seed=seed, options=options)
            return (obs, infos)

        @override(MultiAgentEnv)
        def step(self, action_dict):
            if False:
                for i in range(10):
                    print('nop')
            (obs, rew, terminated, truncated, info) = ({}, {}, {}, {}, {})
            if len(action_dict) == 0:
                raise ValueError('The environment is expecting action for at least one agent.')
            for (i, action) in action_dict.items():
                (obs[i], rew[i], terminated[i], truncated[i], info[i]) = self.envs[i].step(action)
                if terminated[i]:
                    self.terminateds.add(i)
                if truncated[i]:
                    self.truncateds.add(i)
            terminated['__all__'] = len(self.terminateds) + len(self.truncateds) == len(self.envs)
            truncated['__all__'] = len(self.truncateds) == len(self.envs)
            return (obs, rew, terminated, truncated, info)

        @override(MultiAgentEnv)
        def render(self):
            if False:
                return 10
            return self.envs[0].render(self.render_mode)
    return MultiEnv

@PublicAPI
class MultiAgentEnvWrapper(BaseEnv):
    """Internal adapter of MultiAgentEnv to BaseEnv.

    This also supports vectorization if num_envs > 1.
    """

    def __init__(self, make_env: Callable[[int], EnvType], existing_envs: List['MultiAgentEnv'], num_envs: int, restart_failed_sub_environments: bool=False):
        if False:
            i = 10
            return i + 15
        'Wraps MultiAgentEnv(s) into the BaseEnv API.\n\n        Args:\n            make_env: Factory that produces a new MultiAgentEnv instance taking the\n                vector index as only call argument.\n                Must be defined, if the number of existing envs is less than num_envs.\n            existing_envs: List of already existing multi-agent envs.\n            num_envs: Desired num multiagent envs to have at the end in\n                total. This will include the given (already created)\n                `existing_envs`.\n            restart_failed_sub_environments: If True and any sub-environment (within\n                this vectorized env) throws any error during env stepping, we will try\n                to restart the faulty sub-environment. This is done\n                without disturbing the other (still intact) sub-environments.\n        '
        self.make_env = make_env
        self.envs = existing_envs
        self.num_envs = num_envs
        self.restart_failed_sub_environments = restart_failed_sub_environments
        self.terminateds = set()
        self.truncateds = set()
        while len(self.envs) < self.num_envs:
            self.envs.append(self.make_env(len(self.envs)))
        for env in self.envs:
            assert isinstance(env, MultiAgentEnv)
        self._init_env_state(idx=None)
        self._unwrapped_env = self.envs[0].unwrapped

    @override(BaseEnv)
    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        if False:
            print('Hello World!')
        (obs, rewards, terminateds, truncateds, infos) = ({}, {}, {}, {}, {})
        for (i, env_state) in enumerate(self.env_states):
            (obs[i], rewards[i], terminateds[i], truncateds[i], infos[i]) = env_state.poll()
        return (obs, rewards, terminateds, truncateds, infos, {})

    @override(BaseEnv)
    def send_actions(self, action_dict: MultiEnvDict) -> None:
        if False:
            return 10
        for (env_id, agent_dict) in action_dict.items():
            if env_id in self.terminateds or env_id in self.truncateds:
                raise ValueError(f'Env {env_id} is already done and cannot accept new actions')
            env = self.envs[env_id]
            try:
                (obs, rewards, terminateds, truncateds, infos) = env.step(agent_dict)
            except Exception as e:
                if self.restart_failed_sub_environments:
                    logger.exception(e.args[0])
                    self.try_restart(env_id=env_id)
                    obs = e
                    rewards = {}
                    terminateds = {'__all__': True}
                    truncateds = {'__all__': False}
                    infos = {}
                else:
                    raise e
            assert isinstance(obs, (dict, Exception)), 'Not a multi-agent obs dict or an Exception!'
            assert isinstance(rewards, dict), 'Not a multi-agent reward dict!'
            assert isinstance(terminateds, dict), 'Not a multi-agent terminateds dict!'
            assert isinstance(truncateds, dict), 'Not a multi-agent truncateds dict!'
            assert isinstance(infos, dict), 'Not a multi-agent info dict!'
            if isinstance(obs, dict):
                info_diff = set(infos).difference(set(obs))
                if info_diff and info_diff != {'__common__'}:
                    raise ValueError("Key set for infos must be a subset of obs (plus optionally the '__common__' key for infos concerning all/no agents): {} vs {}".format(infos.keys(), obs.keys()))
            if '__all__' not in terminateds:
                raise ValueError("In multi-agent environments, '__all__': True|False must be included in the 'terminateds' dict: got {}.".format(terminateds))
            elif '__all__' not in truncateds:
                raise ValueError("In multi-agent environments, '__all__': True|False must be included in the 'truncateds' dict: got {}.".format(truncateds))
            if terminateds['__all__']:
                self.terminateds.add(env_id)
            if truncateds['__all__']:
                self.truncateds.add(env_id)
            self.env_states[env_id].observe(obs, rewards, terminateds, truncateds, infos)

    @override(BaseEnv)
    def try_reset(self, env_id: Optional[EnvID]=None, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Optional[Tuple[MultiEnvDict, MultiEnvDict]]:
        if False:
            return 10
        ret_obs = {}
        ret_infos = {}
        if isinstance(env_id, int):
            env_id = [env_id]
        if env_id is None:
            env_id = list(range(len(self.envs)))
        for idx in env_id:
            (obs, infos) = self.env_states[idx].reset(seed=seed, options=options)
            if isinstance(obs, Exception):
                if self.restart_failed_sub_environments:
                    self.env_states[idx].env = self.envs[idx] = self.make_env(idx)
                else:
                    raise obs
            else:
                assert isinstance(obs, dict), 'Not a multi-agent obs dict!'
            if obs is not None:
                if idx in self.terminateds:
                    self.terminateds.remove(idx)
                if idx in self.truncateds:
                    self.truncateds.remove(idx)
            ret_obs[idx] = obs
            ret_infos[idx] = infos
        return (ret_obs, ret_infos)

    @override(BaseEnv)
    def try_restart(self, env_id: Optional[EnvID]=None) -> None:
        if False:
            while True:
                i = 10
        if isinstance(env_id, int):
            env_id = [env_id]
        if env_id is None:
            env_id = list(range(len(self.envs)))
        for idx in env_id:
            logger.warning(f'Trying to restart sub-environment at index {idx}.')
            self.env_states[idx].env = self.envs[idx] = self.make_env(idx)
            logger.warning(f'Sub-environment at index {idx} restarted successfully.')

    @override(BaseEnv)
    def get_sub_environments(self, as_dict: bool=False) -> Union[Dict[str, EnvType], List[EnvType]]:
        if False:
            return 10
        if as_dict:
            return {_id: env_state.env for (_id, env_state) in enumerate(self.env_states)}
        return [state.env for state in self.env_states]

    @override(BaseEnv)
    def try_render(self, env_id: Optional[EnvID]=None) -> None:
        if False:
            return 10
        if env_id is None:
            env_id = 0
        assert isinstance(env_id, int)
        return self.envs[env_id].render()

    @property
    @override(BaseEnv)
    @PublicAPI
    def observation_space(self) -> gym.spaces.Dict:
        if False:
            return 10
        return self.envs[0].observation_space

    @property
    @override(BaseEnv)
    @PublicAPI
    def action_space(self) -> gym.Space:
        if False:
            while True:
                i = 10
        return self.envs[0].action_space

    @override(BaseEnv)
    def observation_space_contains(self, x: MultiEnvDict) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return all((self.envs[0].observation_space_contains(val) for val in x.values()))

    @override(BaseEnv)
    def action_space_contains(self, x: MultiEnvDict) -> bool:
        if False:
            i = 10
            return i + 15
        return all((self.envs[0].action_space_contains(val) for val in x.values()))

    @override(BaseEnv)
    def observation_space_sample(self, agent_ids: list=None) -> MultiEnvDict:
        if False:
            while True:
                i = 10
        return {0: self.envs[0].observation_space_sample(agent_ids)}

    @override(BaseEnv)
    def action_space_sample(self, agent_ids: list=None) -> MultiEnvDict:
        if False:
            for i in range(10):
                print('nop')
        return {0: self.envs[0].action_space_sample(agent_ids)}

    @override(BaseEnv)
    def get_agent_ids(self) -> Set[AgentID]:
        if False:
            print('Hello World!')
        return self.envs[0].get_agent_ids()

    def _init_env_state(self, idx: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        "Resets all or one particular sub-environment's state (by index).\n\n        Args:\n            idx: The index to reset at. If None, reset all the sub-environments' states.\n        "
        if idx is None:
            self.env_states = [_MultiAgentEnvState(env, self.restart_failed_sub_environments) for env in self.envs]
        else:
            assert isinstance(idx, int)
            self.env_states[idx] = _MultiAgentEnvState(self.envs[idx], self.restart_failed_sub_environments)

class _MultiAgentEnvState:

    def __init__(self, env: MultiAgentEnv, return_error_as_obs: bool=False):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(env, MultiAgentEnv)
        self.env = env
        self.return_error_as_obs = return_error_as_obs
        self.initialized = False
        self.last_obs = {}
        self.last_rewards = {}
        self.last_terminateds = {'__all__': False}
        self.last_truncateds = {'__all__': False}
        self.last_infos = {}

    def poll(self) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        if False:
            return 10
        if not self.initialized:
            self.reset()
            self.initialized = True
        observations = self.last_obs
        rewards = {}
        terminateds = {'__all__': self.last_terminateds['__all__']}
        truncateds = {'__all__': self.last_truncateds['__all__']}
        infos = self.last_infos
        if terminateds['__all__'] or truncateds['__all__'] or isinstance(observations, Exception):
            rewards = self.last_rewards
            self.last_rewards = {}
            terminateds = self.last_terminateds
            if isinstance(observations, Exception):
                terminateds['__all__'] = True
                truncateds['__all__'] = False
            self.last_terminateds = {}
            truncateds = self.last_truncateds
            self.last_truncateds = {}
            self.last_obs = {}
            infos = self.last_infos
            self.last_infos = {}
        else:
            for ag in observations.keys():
                if ag in self.last_rewards:
                    rewards[ag] = self.last_rewards[ag]
                    del self.last_rewards[ag]
                if ag in self.last_terminateds:
                    terminateds[ag] = self.last_terminateds[ag]
                    del self.last_terminateds[ag]
                if ag in self.last_truncateds:
                    truncateds[ag] = self.last_truncateds[ag]
                    del self.last_truncateds[ag]
        self.last_terminateds['__all__'] = False
        self.last_truncateds['__all__'] = False
        return (observations, rewards, terminateds, truncateds, infos)

    def observe(self, obs: MultiAgentDict, rewards: MultiAgentDict, terminateds: MultiAgentDict, truncateds: MultiAgentDict, infos: MultiAgentDict):
        if False:
            return 10
        self.last_obs = obs
        for (ag, r) in rewards.items():
            if ag in self.last_rewards:
                self.last_rewards[ag] += r
            else:
                self.last_rewards[ag] = r
        for (ag, d) in terminateds.items():
            if ag in self.last_terminateds:
                self.last_terminateds[ag] = self.last_terminateds[ag] or d
            else:
                self.last_terminateds[ag] = d
        for (ag, t) in truncateds.items():
            if ag in self.last_truncateds:
                self.last_truncateds[ag] = self.last_truncateds[ag] or t
            else:
                self.last_truncateds[ag] = t
        self.last_infos = infos

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        if False:
            i = 10
            return i + 15
        try:
            obs_and_infos = self.env.reset(seed=seed, options=options)
        except Exception as e:
            if self.return_error_as_obs:
                logger.exception(e.args[0])
                obs_and_infos = (e, e)
            else:
                raise e
        (self.last_obs, self.last_infos) = obs_and_infos
        self.last_rewards = {}
        self.last_terminateds = {'__all__': False}
        self.last_truncateds = {'__all__': False}
        return (self.last_obs, self.last_infos)