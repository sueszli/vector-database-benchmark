from collections import OrderedDict
import gymnasium as gym
from typing import Dict, List, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID
GROUP_REWARDS = '_group_rewards'
GROUP_INFO = '_group_info'

@DeveloperAPI
class GroupAgentsWrapper(MultiAgentEnv):
    """Wraps a MultiAgentEnv environment with agents grouped as specified.

    See multi_agent_env.py for the specification of groups.

    This API is experimental.
    """

    def __init__(self, env: MultiAgentEnv, groups: Dict[str, List[AgentID]], obs_space: Optional[gym.Space]=None, act_space: Optional[gym.Space]=None):
        if False:
            return 10
        'Wrap an existing MultiAgentEnv to group agent ID together.\n\n        See `MultiAgentEnv.with_agent_groups()` for more detailed usage info.\n\n        Args:\n            env: The env to wrap and whose agent IDs to group into new agents.\n            groups: Mapping from group id to a list of the agent ids\n                of group members. If an agent id is not present in any group\n                value, it will be left ungrouped. The group id becomes a new agent ID\n                in the final environment.\n            obs_space: Optional observation space for the grouped\n                env. Must be a tuple space. If not provided, will infer this to be a\n                Tuple of n individual agents spaces (n=num agents in a group).\n            act_space: Optional action space for the grouped env.\n                Must be a tuple space. If not provided, will infer this to be a Tuple\n                of n individual agents spaces (n=num agents in a group).\n        '
        super().__init__()
        self.env = env
        if hasattr(self.env, '_skip_env_checking'):
            self._skip_env_checking = self.env._skip_env_checking
        self.groups = groups
        self.agent_id_to_group = {}
        for (group_id, agent_ids) in groups.items():
            for agent_id in agent_ids:
                if agent_id in self.agent_id_to_group:
                    raise ValueError('Agent id {} is in multiple groups'.format(agent_id))
                self.agent_id_to_group[agent_id] = group_id
        if obs_space is not None:
            self.observation_space = obs_space
        if act_space is not None:
            self.action_space = act_space
        for group_id in groups.keys():
            self._agent_ids.add(group_id)

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        if False:
            i = 10
            return i + 15
        (obs, info) = self.env.reset(seed=seed, options=options)
        return (self._group_items(obs), self._group_items(info, agg_fn=lambda gvals: {GROUP_INFO: list(gvals.values())}))

    def step(self, action_dict):
        if False:
            for i in range(10):
                print('nop')
        action_dict = self._ungroup_items(action_dict)
        (obs, rewards, terminateds, truncateds, infos) = self.env.step(action_dict)
        obs = self._group_items(obs)
        rewards = self._group_items(rewards, agg_fn=lambda gvals: list(gvals.values()))
        terminateds = self._group_items(terminateds, agg_fn=lambda gvals: all(gvals.values()))
        truncateds = self._group_items(truncateds, agg_fn=lambda gvals: all(gvals.values()))
        infos = self._group_items(infos, agg_fn=lambda gvals: {GROUP_INFO: list(gvals.values())})
        for (agent_id, rew) in rewards.items():
            if isinstance(rew, list):
                rewards[agent_id] = sum(rew)
                if agent_id not in infos:
                    infos[agent_id] = {}
                infos[agent_id][GROUP_REWARDS] = rew
        return (obs, rewards, terminateds, truncateds, infos)

    def _ungroup_items(self, items):
        if False:
            while True:
                i = 10
        out = {}
        for (agent_id, value) in items.items():
            if agent_id in self.groups:
                assert len(value) == len(self.groups[agent_id]), (agent_id, value, self.groups)
                for (a, v) in zip(self.groups[agent_id], value):
                    out[a] = v
            else:
                out[agent_id] = value
        return out

    def _group_items(self, items, agg_fn=lambda gvals: list(gvals.values())):
        if False:
            for i in range(10):
                print('nop')
        grouped_items = {}
        for (agent_id, item) in items.items():
            if agent_id in self.agent_id_to_group:
                group_id = self.agent_id_to_group[agent_id]
                if group_id in grouped_items:
                    continue
                group_out = OrderedDict()
                for a in self.groups[group_id]:
                    if a in items:
                        group_out[a] = items[a]
                    else:
                        raise ValueError('Missing member of group {}: {}: {}'.format(group_id, a, items))
                grouped_items[group_id] = agg_fn(group_out)
            else:
                grouped_items[agent_id] = item
        return grouped_items