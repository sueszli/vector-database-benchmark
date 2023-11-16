import uuid
import gymnasium as gym
from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.env.external_env import ExternalEnv, _ExternalEnvEpisode
from ray.rllib.utils.typing import MultiAgentDict

@PublicAPI
class ExternalMultiAgentEnv(ExternalEnv):
    """This is the multi-agent version of ExternalEnv."""

    @PublicAPI
    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        if False:
            for i in range(10):
                print('nop')
        'Initializes an ExternalMultiAgentEnv instance.\n\n        Args:\n            action_space: Action space of the env.\n            observation_space: Observation space of the env.\n        '
        ExternalEnv.__init__(self, action_space, observation_space)
        if isinstance(self.action_space, dict) or isinstance(self.observation_space, dict):
            if not self.action_space.keys() == self.observation_space.keys():
                raise ValueError('Agent ids disagree for action space and obs space dict: {} {}'.format(self.action_space.keys(), self.observation_space.keys()))

    @PublicAPI
    def run(self):
        if False:
            print('Hello World!')
        'Override this to implement the multi-agent run loop.\n\n        Your loop should continuously:\n            1. Call self.start_episode(episode_id)\n            2. Call self.get_action(episode_id, obs_dict)\n                    -or-\n                    self.log_action(episode_id, obs_dict, action_dict)\n            3. Call self.log_returns(episode_id, reward_dict)\n            4. Call self.end_episode(episode_id, obs_dict)\n            5. Wait if nothing to do.\n\n        Multiple episodes may be started at the same time.\n        '
        raise NotImplementedError

    @PublicAPI
    @override(ExternalEnv)
    def start_episode(self, episode_id: Optional[str]=None, training_enabled: bool=True) -> str:
        if False:
            i = 10
            return i + 15
        if episode_id is None:
            episode_id = uuid.uuid4().hex
        if episode_id in self._finished:
            raise ValueError('Episode {} has already completed.'.format(episode_id))
        if episode_id in self._episodes:
            raise ValueError('Episode {} is already started'.format(episode_id))
        self._episodes[episode_id] = _ExternalEnvEpisode(episode_id, self._results_avail_condition, training_enabled, multiagent=True)
        return episode_id

    @PublicAPI
    @override(ExternalEnv)
    def get_action(self, episode_id: str, observation_dict: MultiAgentDict) -> MultiAgentDict:
        if False:
            for i in range(10):
                print('nop')
        'Record an observation and get the on-policy action.\n\n        Thereby, observation_dict is expected to contain the observation\n        of all agents acting in this episode step.\n\n        Args:\n            episode_id: Episode id returned from start_episode().\n            observation_dict: Current environment observation.\n\n        Returns:\n            action: Action from the env action space.\n        '
        episode = self._get(episode_id)
        return episode.wait_for_action(observation_dict)

    @PublicAPI
    @override(ExternalEnv)
    def log_action(self, episode_id: str, observation_dict: MultiAgentDict, action_dict: MultiAgentDict) -> None:
        if False:
            i = 10
            return i + 15
        'Record an observation and (off-policy) action taken.\n\n        Args:\n            episode_id: Episode id returned from start_episode().\n            observation_dict: Current environment observation.\n            action_dict: Action for the observation.\n        '
        episode = self._get(episode_id)
        episode.log_action(observation_dict, action_dict)

    @PublicAPI
    @override(ExternalEnv)
    def log_returns(self, episode_id: str, reward_dict: MultiAgentDict, info_dict: MultiAgentDict=None, multiagent_done_dict: MultiAgentDict=None) -> None:
        if False:
            i = 10
            return i + 15
        'Record returns from the environment.\n\n        The reward will be attributed to the previous action taken by the\n        episode. Rewards accumulate until the next action. If no reward is\n        logged before the next action, a reward of 0.0 is assumed.\n\n        Args:\n            episode_id: Episode id returned from start_episode().\n            reward_dict: Reward from the environment agents.\n            info_dict: Optional info dict.\n            multiagent_done_dict: Optional done dict for agents.\n        '
        episode = self._get(episode_id)
        for (agent, rew) in reward_dict.items():
            if agent in episode.cur_reward_dict:
                episode.cur_reward_dict[agent] += rew
            else:
                episode.cur_reward_dict[agent] = rew
        if multiagent_done_dict:
            for (agent, done) in multiagent_done_dict.items():
                episode.cur_done_dict[agent] = done
        if info_dict:
            episode.cur_info_dict = info_dict or {}

    @PublicAPI
    @override(ExternalEnv)
    def end_episode(self, episode_id: str, observation_dict: MultiAgentDict) -> None:
        if False:
            while True:
                i = 10
        'Record the end of an episode.\n\n        Args:\n            episode_id: Episode id returned from start_episode().\n            observation_dict: Current environment observation.\n        '
        episode = self._get(episode_id)
        self._finished.add(episode.episode_id)
        episode.done(observation_dict)