import numpy as np
import uuid
from typing import Any, Dict, List, Optional, Union
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict

class MultiAgentEpisode:
    """Stores multi-agent episode data.

    The central attribute of the class is the timestep mapping
    `global_t_to_local_t` that maps the global (environment)
    timestep to the local (agent) timesteps.

    The `MultiAgentEpisode` is based on the `SingleAgentEpisode`s
    for each agent, stored in `MultiAgentEpisode.agent_episodes`.
    """

    def __init__(self, id_: Optional[str]=None, agent_ids: List[str]=None, agent_episode_ids: Optional[Dict[str, str]]=None, *, observations: Optional[List[MultiAgentDict]]=None, actions: Optional[List[MultiAgentDict]]=None, rewards: Optional[List[MultiAgentDict]]=None, states: Optional[List[MultiAgentDict]]=None, infos: Optional[List[MultiAgentDict]]=None, t_started: int=0, is_terminated: Optional[bool]=False, is_truncated: Optional[bool]=False, render_images: Optional[List[np.ndarray]]=None, extra_model_outputs: Optional[List[MultiAgentDict]]=None) -> 'MultiAgentEpisode':
        if False:
            for i in range(10):
                print('nop')
        "Initializes a `MultiAgentEpisode`.\n\n        Args:\n            id_: Optional. Either a string to identify an episode or None.\n                If None, a hexadecimal id is created. In case of providing\n                a string, make sure that it is unique, as episodes get\n                concatenated via this string.\n            agent_ids: Obligatory. A list of strings containing the agent ids.\n                These have to be provided at initialization.\n            agent_episode_ids: Optional. Either a dictionary mapping agent ids\n                corresponding `SingleAgentEpisode` or None. If None, each\n                `SingleAgentEpisode` in `MultiAgentEpisode.agent_episodes`\n                will generate a hexadecimal code. If a dictionary is provided\n                make sure that ids are unique as agents'  `SingleAgentEpisode`s\n                get concatenated or recreated by it.\n            observations: A dictionary mapping from agent ids to observations.\n                Can be None. If provided, it should be provided together with\n                all other episode data (actions, rewards, etc.)\n            actions: A dictionary mapping from agent ids to corresponding actions.\n                Can be None. If provided, it should be provided together with\n                all other episode data (observations, rewards, etc.).\n            rewards: A dictionary mapping from agent ids to corresponding rewards.\n                Can be None. If provided, it should be provided together with\n                all other episode data (observations, rewards, etc.).\n            infos: A dictionary mapping from agent ids to corresponding infos.\n                Can be None. If provided, it should be provided together with\n                all other episode data (observations, rewards, etc.).\n            states: A dictionary mapping from agent ids to their corresponding\n                modules' hidden states. These will be stored into the\n                `SingleAgentEpisode`s in `MultiAgentEpisode.agent_episodes`.\n                Can be None.\n            t_started: Optional. An unsigned int that defines the starting point\n                of the episode. This is only different from zero, if an ongoing\n                episode is created.\n            is_terminazted: Optional. A boolean defining, if an environment has\n                terminated. The default is `False`, i.e. the episode is ongoing.\n            is_truncated: Optional. A boolean, defining, if an environment is\n                truncated. The default is `False`, i.e. the episode is ongoing.\n            render_images: Optional. A list of RGB uint8 images from rendering\n                the environment.\n            extra_model_outputs: Optional. A dictionary mapping agent ids to their\n                corresponding extra model outputs. Each of the latter is a list of\n                dictionaries containing specific model outputs for the algorithm\n                used (e.g. `vf_preds` and `action_logp` for PPO) from a rollout.\n                If data is provided it should be complete (i.e. observations,\n                actions, rewards, is_terminated, is_truncated, and all necessary\n                `extra_model_outputs`).\n        "
        self.id_: str = id_ or uuid.uuid4().hex
        self._agent_ids: Union[List[str], List[object]] = [] if agent_ids is None else agent_ids
        self.t = self.t_started = t_started if t_started is not None else max(len(observations) - 1, 0)
        self.global_t_to_local_t: Dict[str, List[int]] = self._generate_ts_mapping(observations)
        self.agent_episodes: MultiAgentDict = {agent_id: self._generate_single_agent_episode(agent_id, agent_episode_ids, observations, actions, rewards, infos, states, extra_model_outputs) for agent_id in self._agent_ids}
        self.is_terminated: bool = is_terminated
        self.is_truncated: bool = is_truncated
        assert render_images is None or observations is not None
        self.render_images: Union[List[np.ndarray], List[object]] = [] if render_images is None else render_images

    def concat_episode(self, episode_chunk: 'MultiAgentEpisode') -> None:
        if False:
            while True:
                i = 10
        'Adds the given `episode_chunk` to the right side of self.\n\n        For concatenating episodes the following rules hold:\n            - IDs are identical.\n            - timesteps match (`t` of `self` matches `t_started` of `episode_chunk`).\n\n        Args:\n            episode_chunk: `MultiAgentEpsiode` instance that should be concatenated\n                to `self`.\n        '
        assert episode_chunk.id_ == self.id_
        assert not self.is_done
        assert self.t == episode_chunk.t_started
        observations: MultiAgentDict = self.get_observations()
        for (agent_id, agent_obs) in episode_chunk.get_observations(indices=0):
            assert agent_id in observations
            assert observations[agent_id] == agent_obs
        for agent_id in observations:
            self.agent_episodes[agent_id].observations.pop()
        for (agent_id, agent_eps) in self.agent_episodes:
            agent_eps[agent_id].concat_episode(episode_chunk.agent_episodes[agent_id])
            self.global_t_to_local_t[agent_id][:-1] += episode_chunk.global_t_to_local_t[agent_id]
        self.t = episode_chunk.t
        if episode_chunk.is_terminated:
            self.is_terminated = True
        if episode_chunk.is_truncated:
            self.is_truncated = True

    def get_observations(self, indices: Union[int, List[int]]=-1, global_ts: bool=True) -> MultiAgentDict:
        if False:
            print('Hello World!')
        'Gets observations for all agents that stepped in the last timesteps.\n\n        Note that observations are only returned for agents that stepped\n        during the given index range.\n\n        Args:\n            indices: Either a single index or a list of indices. The indices\n                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).\n                This defines the time indices for which the observations\n                should be returned.\n            global_ts: Boolean that defines, if the indices should be considered\n                environment (`True`) or agent (`False`) steps.\n\n        Returns: A dictionary mapping agent ids to observations (of different\n            timesteps). Only for agents that have stepped (were ready) at a\n            timestep, observations are returned (i.e. not all agent ids are\n            necessarily in the keys).\n        '
        return self._getattr_by_index('observations', indices, global_ts)

    def get_actions(self, indices: Union[int, List[int]]=-1, global_ts: bool=True) -> MultiAgentDict:
        if False:
            while True:
                i = 10
        'Gets actions for all agents that stepped in the last timesteps.\n\n        Note that actions are only returned for agents that stepped\n        during the given index range.\n\n        Args:\n            indices: Either a single index or a list of indices. The indices\n                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).\n                This defines the time indices for which the actions\n                should be returned.\n            global_ts: Boolean that defines, if the indices should be considered\n                environment (`True`) or agent (`False`) steps.\n\n        Returns: A dictionary mapping agent ids to actions (of different\n            timesteps). Only for agents that have stepped (were ready) at a\n            timestep, actions are returned (i.e. not all agent ids are\n            necessarily in the keys).\n        '
        return self._getattr_by_index('actions', indices, global_ts)

    def get_rewards(self, indices: Union[int, List[int]]=-1, global_ts: bool=True) -> MultiAgentDict:
        if False:
            print('Hello World!')
        'Gets rewards for all agents that stepped in the last timesteps.\n\n        Note that rewards are only returned for agents that stepped\n        during the given index range.\n\n        Args:\n            indices: Either a single index or a list of indices. The indices\n                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).\n                This defines the time indices for which the rewards\n                should be returned.\n            global_ts: Boolean that defines, if the indices should be considered\n                environment (`True`) or agent (`False`) steps.\n\n        Returns: A dictionary mapping agent ids to rewards (of different\n            timesteps). Only for agents that have stepped (were ready) at a\n            timestep, rewards are returned (i.e. not all agent ids are\n            necessarily in the keys).\n        '
        return self._getattr_by_index('rewards', indices, global_ts)

    def get_infos(self, indices: Union[int, List[int]]=-1, global_ts: bool=True) -> MultiAgentDict:
        if False:
            for i in range(10):
                print('nop')
        'Gets infos for all agents that stepped in the last timesteps.\n\n        Note that infos are only returned for agents that stepped\n        during the given index range.\n\n        Args:\n            indices: Either a single index or a list of indices. The indices\n                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).\n                This defines the time indices for which the infos\n                should be returned.\n            global_ts: Boolean that defines, if the indices should be considered\n                environment (`True`) or agent (`False`) steps.\n\n        Returns: A dictionary mapping agent ids to infos (of different\n            timesteps). Only for agents that have stepped (were ready) at a\n            timestep, infos are returned (i.e. not all agent ids are\n            necessarily in the keys).\n        '
        return self._getattr_by_index('infos', indices, global_ts)

    def get_extra_model_outputs(self, indices: Union[int, List[int]]=-1, global_ts: bool=True) -> MultiAgentDict:
        if False:
            for i in range(10):
                print('nop')
        'Gets extra model outputs for all agents that stepped in the last timesteps.\n\n        Note that extra model outputs are only returned for agents that stepped\n        during the given index range.\n\n        Args:\n            indices: Either a single index or a list of indices. The indices\n                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).\n                This defines the time indices for which the extra model outputs.\n                should be returned.\n            global_ts: Boolean that defines, if the indices should be considered\n                environment (`True`) or agent (`False`) steps.\n\n        Returns: A dictionary mapping agent ids to extra model outputs (of different\n            timesteps). Only for agents that have stepped (were ready) at a\n            timestep, extra model outputs are returned (i.e. not all agent ids are\n            necessarily in the keys).\n        '
        return self._getattr_by_index('extra_model_outputs', indices, global_ts)

    def add_initial_observation(self, *, initial_observation: MultiAgentDict, initial_info: Optional[MultiAgentDict]=None, initial_state: Optional[MultiAgentDict]=None, initial_render_image: Optional[np.ndarray]=None) -> None:
        if False:
            print('Hello World!')
        'Stores initial observation.\n\n        Args:\n            initial_observation: Obligatory. A dictionary, mapping agent ids\n                to initial observations. Note that not all agents must have\n                an initial observation.\n            initial_info: Optional. A dictionary, mapping agent ids to initial\n                infos. Note that not all agents must have an initial info.\n            initial_state: Optional. A dictionary, mapping agent ids to the\n                initial hidden states of their corresponding model (`RLModule`).\n                Note, this is only available, if the models are stateful. Note\n                also that not all agents must have an initial state at `t=0`.\n            initial_render_image: An RGB uint8 image from rendering the\n                environment.\n        '
        assert not self.is_done
        assert self.t == self.t_started == 0
        if len(self.global_t_to_local_t) == 0:
            self.global_t_to_local_t = {agent_id: [] for agent_id in self._agent_ids}
        if initial_render_image is not None:
            self.render_images.append(initial_render_image)
        for agent_id in initial_observation.keys():
            self.global_t_to_local_t[agent_id].append(self.t)
            self.agent_episodes[agent_id].add_initial_observation(initial_observation=initial_observation[agent_id], initial_info=None if initial_info is None else initial_info[agent_id], initial_state=None if initial_state is None else initial_state[agent_id])

    def add_timestep(self, observation: MultiAgentDict, action: MultiAgentDict, reward: MultiAgentDict, *, info: Optional[MultiAgentDict]=None, state: Optional[MultiAgentDict]=None, is_terminated: Optional[bool]=None, is_truncated: Optional[bool]=None, render_image: Optional[np.ndarray]=None, extra_model_output: Optional[MultiAgentDict]=None) -> None:
        if False:
            print('Hello World!')
        'Adds a timestep to the episode.\n\n        Args:\n            observation: Mandatory. A dictionary, mapping agent ids to their\n                corresponding observations. Note that not all agents must have stepped\n                a this timestep.\n            action: Mandatory. A dictionary, mapping agent ids to their\n                corresponding actions. Note that not all agents must have stepped\n                a this timestep.\n            reward: Mandatory. A dictionary, mapping agent ids to their\n                corresponding observations. Note that not all agents must have stepped\n                a this timestep.\n            info: Optional. A dictionary, mapping agent ids to their\n                corresponding info. Note that not all agents must have stepped\n                a this timestep.\n            state: Optional. A dictionary, mapping agent ids to their\n                corresponding hidden model state. Note, this is only available for a\n                stateful model. Also note that not all agents must have stepped a this\n                timestep.\n            is_terminated: A boolean indicating, if the environment has been\n                terminated.\n            is_truncated: A boolean indicating, if the environment has been\n                truncated.\n            render_image: Optional. An RGB uint8 image from rendering the environment.\n            extra_model_output: Optional. A dictionary, mapping agent ids to their\n                corresponding specific model outputs (also in a dictionary; e.g.\n                `vf_preds` for PPO).\n        '
        assert not self.is_done
        self.t += 1
        self.is_terminated = False if is_terminated is None else is_terminated['__all__']
        self.is_truncated = False if is_truncated is None else is_truncated['__all__']
        for agent_id in observation.keys():
            self.global_t_to_local_t[agent_id].append(self.t)
            if render_image is not None:
                self.render_images.append(render_image)
            self.agent_episodes[agent_id].add_timestep(observation[agent_id], action[agent_id], reward[agent_id], info=None if info is None else info[agent_id], state=None if state is None else state[agent_id], is_terminated=None if is_terminated is None else is_terminated[agent_id], is_truncated=None if is_truncated is None else is_truncated[agent_id], render_image=None if render_image is None else render_image[agent_id], extra_model_output=None if extra_model_output is None else extra_model_output[agent_id])

    @property
    def is_done(self):
        if False:
            while True:
                i = 10
        "Whether the episode is actually done (terminated or truncated).\n\n        A done episode cannot be continued via `self.add_timestep()` or being\n        concatenated on its right-side with another episode chunk or being\n        succeeded via `self.create_successor()`.\n\n        Note that in a multi-agent environment this does not necessarily\n        correspond to single agents having terminated or being truncated.\n\n        `self.is_terminated` should be `True`, if all agents are terminated and\n        `self.is_truncated` should be `True`, if all agents are truncated. If\n        only one or more (but not all!) agents are `terminated/truncated the\n        `MultiAgentEpisode.is_terminated/is_truncated` should be `False`. This\n        information about single agent's terminated/truncated states can always\n        be retrieved from the `SingleAgentEpisode`s inside the 'MultiAgentEpisode`\n        one.\n\n        If all agents are either terminated or truncated, but in a mixed fashion,\n        i.e. some are terminated and others are truncated: This is currently\n        undefined and could potentially be a problem (if a user really implemented\n        such a multi-agent env that behaves this way).\n\n        Returns:\n            Boolean defining if an episode has either terminated or truncated.\n        "
        return self.is_terminated or self.is_truncated

    def create_successor(self) -> 'MultiAgentEpisode':
        if False:
            print('Hello World!')
        'Restarts an ongoing episode from its last observation.\n\n        Note, this method is used so far specifically for the case of\n        `batch_mode="truncated_episodes"` to ensure that episodes are\n        immutable inside the `EnvRunner` when truncated and passed over\n        to postprocessing.\n\n        The newly created `MultiAgentEpisode` contains the same id, and\n        starts at the timestep where it\'s predecessor stopped in the last\n        rollout. Last observations, infos, rewards, etc. are carried over\n        from the predecessor. This also helps to not carry stale data that\n        had been collected in the last rollout when rolling out the new\n        policy in the next iteration (rollout).\n\n        Returns: A MultiAgentEpisode starting at the timepoint where\n            its predecessor stopped.\n        '
        assert not self.is_done
        observations = self.get_observations()
        infos = self.get_infos()
        return MultiAgentEpisode(id=self.id_, agent_episode_ids={agent_id: agent_eps.id_ for (agent_id, agent_eps) in self.agent_episodes}, observations=observations, infos=infos, t_started=self.t)

    def get_state(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "Returns the state of a multi-agent episode.\n\n        Note that from an episode's state the episode itself can\n        be recreated.\n\n        Returns: A dicitonary containing pickable data fro a\n            `MultiAgentEpisode`.\n        "
        return list({'id_': self.id_, 'agent_ids': self._agent_ids, 'global_t_to_local_t': self.global_t_to_local_t, 'agent_episodes': list({agent_id: agent_eps.get_state() for (agent_id, agent_eps) in self.agent_episodes.items()}.items()), 't_started': self.t_started, 't': self.t, 'is_terminated': self.is_terminated, 'is_truncated': self.is_truncated}.items())

    @staticmethod
    def from_state(state) -> None:
        if False:
            return 10
        'Creates a multi-agent episode from a state dictionary.\n\n        See `MultiAgentEpisode.get_state()` for creating a state for\n        a `MultiAgentEpisode` pickable state. For recreating a\n        `MultiAgentEpisode` from a state, this state has to be complete,\n        i.e. all data must have been stored in the state.\n        '
        eps = MultiAgentEpisode(id=state[0][1])
        eps._agent_ids = state[1][1]
        eps.global_t_to_local_t = state[2][1]
        eps.agent_episodes = {agent_id: SingleAgentEpisode.from_state(agent_state) for (agent_id, agent_state) in state[3][1]}
        eps.t_started = state[3][1]
        eps.t = state[4][1]
        eps.is_terminated = state[5][1]
        eps.is_trcunated = state[6][1]
        return eps

    def to_sample_batch(self) -> MultiAgentBatch:
        if False:
            for i in range(10):
                print('nop')
        "Converts a `MultiAgentEpisode` into a `MultiAgentBatch`.\n\n        Each `SingleAgentEpisode` instances in\n        `MultiAgentEpisode.agent_epiosdes` will be converted into\n        a `SampleBatch` and the environment timestep will be passed\n        towards the `MultiAgentBatch`'s `count`.\n\n        Returns: A `MultiAgentBatch` instance.\n        "
        return MultiAgentBatch(policy_batches={agent_id: agent_eps.to_sample_batch() for (agent_id, agent_eps) in self.agent_episodes.items()}, env_steps=self.t)

    def get_return(self) -> float:
        if False:
            print('Hello World!')
        'Get the all-agent return.\n\n        Returns: A float. The aggregate return from all agents.\n        '
        return sum([agent_eps.get_return() for agent_eps in self.agent_episodes.values()])

    def _generate_ts_mapping(self, observations: List[MultiAgentDict]) -> MultiAgentDict:
        if False:
            i = 10
            return i + 15
        'Generates a timestep mapping to local agent timesteps.\n\n        This helps us to keep track of which agent stepped at\n        which global (environment) timestep.\n        Note that the local (agent) timestep is given by the index\n        of the list for each agent.\n\n        Args:\n            observations: A list of observations.Each observations maps agent\n                ids to their corresponding observation.\n\n        Returns: A dictionary mapping agents to time index lists. The latter\n            contain the global (environment) timesteps at which the agent\n            stepped (was ready).\n        '
        if len(self._agent_ids) > 0:
            global_t_to_local_t = {agent: _IndexMapping() for agent in self._agent_ids}
            if len(observations) > 0:
                for (t, agent_map) in enumerate(observations):
                    for agent_id in agent_map:
                        global_t_to_local_t[agent_id].append(t)
            else:
                global_t_to_local_t = {}
        else:
            global_t_to_local_t = {}
        return global_t_to_local_t

    def _generate_single_agent_episode(self, agent_id: str, agent_episode_ids: Optional[Dict[str, str]]=None, observations: Optional[List[MultiAgentDict]]=None, actions: Optional[List[MultiAgentDict]]=None, rewards: Optional[List[MultiAgentDict]]=None, infos: Optional[List[MultiAgentDict]]=None, states: Optional[MultiAgentDict]=None, extra_model_outputs: Optional[MultiAgentDict]=None) -> SingleAgentEpisode:
        if False:
            for i in range(10):
                print('nop')
        "Generates a `SingleAgentEpisode` from multi-agent data.\n\n        Note, if no data is provided an empty `SingleAgentEpiosde`\n        will be returned that starts at `SIngleAgentEpisode.t_started=0`.\n\n        Args:\n            agent_id: String, idnetifying the agent for which the data should\n                be extracted.\n            agent_episode_ids: Optional. A dictionary mapping agents to\n                corresponding episode ids. If `None` the `SingleAgentEpisode`\n                creates a hexadecimal code.\n            observations: Optional. A list of dictionaries, each mapping\n                from agent ids to observations. When data is provided\n                it should be complete, i.e. observations, actions, rewards,\n                etc. should be provided.\n            actions: Optional. A list of dictionaries, each mapping\n                from agent ids to actions. When data is provided\n                it should be complete, i.e. observations, actions, rewards,\n                etc. should be provided.\n            rewards: Optional. A list of dictionaries, each mapping\n                from agent ids to rewards. When data is provided\n                it should be complete, i.e. observations, actions, rewards,\n                etc. should be provided.\n            infos: Optional. A list of dictionaries, each mapping\n                from agent ids to infos. When data is provided\n                it should be complete, i.e. observations, actions, rewards,\n                etc. should be provided.\n            states: Optional. A dicitionary mapping each agent to it's\n                module's hidden model state (if the model is stateful).\n            extra_model_outputs: Optional. A list of agent mappings for every\n                timestep. Each of these dictionaries maps an agent to its\n                corresponding `extra_model_outputs`, which a re specific model\n                outputs needed by the algorithm used (e.g. `vf_preds` and\n                `action_logp` for PPO). f data is provided it should be complete\n                (i.e. observations, actions, rewards, is_terminated, is_truncated,\n                and all necessary `extra_model_outputs`).\n\n        Returns: An instance of `SingleAgentEpisode` containing the agent's\n            extracted episode data.\n        "
        episode_id = None if agent_episode_ids is None else agent_episode_ids[agent_id]
        if len(self.global_t_to_local_t) > 0:
            agent_observations = None if observations is None else self._get_single_agent_data(agent_id, observations)
            agent_actions = None if actions is None else self._get_single_agent_data(agent_id, actions, start_index=1, shift=-1)
            agent_rewards = None if rewards is None else self._get_single_agent_data(agent_id, rewards, start_index=1, shift=-1)
            agent_infos = None if infos is None else self._get_single_agent_data(agent_id, infos, start_index=1)
            agent_states = None if states is None else self._get_single_agent_data(agent_id, states, start_index=1, shift=-1)
            agent_extra_model_outputs = None if extra_model_outputs is None else self._get_single_agent_data(agent_id, extra_model_outputs, start_index=1, shift=-1)
            return SingleAgentEpisode(id_=episode_id, observations=agent_observations, actions=agent_actions, rewards=agent_rewards, infos=agent_infos, states=agent_states, extra_model_outputs=agent_extra_model_outputs)
        else:
            return SingleAgentEpisode(id_=episode_id)

    def _getattr_by_index(self, attr: str='observations', indices: Union[int, List[int]]=-1, global_ts: bool=True) -> MultiAgentDict:
        if False:
            return 10
        if global_ts:
            if isinstance(indices, list):
                indices = [self.t + (idx if idx < 0 else idx) for idx in indices]
            else:
                indices = [self.t + indices] if indices < 0 else [indices]
            return {agent_id: list(map(getattr(agent_eps, attr).__getitem__, self.global_t_to_local_t[agent_id].find_indices(indices))) for (agent_id, agent_eps) in self.agent_episodes.items() if len(self.global_t_to_local_t[agent_id].find_indices(indices)) > 0}
        else:
            if not isinstance(indices, list):
                indices = [indices]
            return {agent_id: list(map(getattr(agent_eps, attr).__getitem__, indices)) for (agent_id, agent_eps) in self.agent_episodes.items() if self.agent_episodes[agent_id].t > 0}

    def _get_single_agent_data(self, agent_id: str, ma_data: List[MultiAgentDict], start_index: int=0, end_index: Optional[int]=None, shift: int=0) -> List[Any]:
        if False:
            i = 10
            return i + 15
        'Returns single agent data from multi-agent data.\n\n        Args:\n            agent_id: A string identifying the agent for which the\n                data should be extracted.\n            ma_data: A List of dictionaries, each containing multi-agent\n                data, i.e. mapping from agent ids to timestep data.\n            start_index: An integer defining the start point of the\n                extration window. The default starts at the beginning of the\n                the `ma_data` list.\n            end_index: Optional. An integer defining the end point of the\n                extraction window. If `None`, the extraction window will be\n                until the end of the `ma_data` list.g\n            shift: An integer that defines by which amount to shift the\n                running index for extraction. This is for example needed\n                when we extract data that started at index 1.\n\n        Returns: A list containing single-agent data from the multi-agent\n            data provided.\n        '
        return [singleton[agent_id] for singleton in list(map(ma_data.__getitem__, [i + shift for i in self.global_t_to_local_t[agent_id][start_index:end_index]])) if agent_id in singleton.keys()]

    def __len__(self):
        if False:
            return 10
        'Returns the length of an `MultiAgentEpisode`.\n\n        Note that the length of an episode is defined by the difference\n        between its actual timestep and the starting point.\n\n        Returns: An integer defining the length of the episode or an\n            error if the episode has not yet started.\n        '
        assert self.t_started < self.t, "ERROR: Cannot determine length of episode that hasn't started, yet!Call `MultiAgentEpisode.add_initial_observation(initial_observation=)` first (after which `len(MultiAgentEpisode)` will be 0)."
        return self.t - self.t_started

class _IndexMapping(list):
    """Provides lists with a method to find multiple elements.

    This class is used for the timestep mapping which is central to
    the multi-agent episode. For each agent the timestep mapping is
    implemented with an `IndexMapping`.

    The `IndexMapping.find_indices` method simplifies the search for
    multiple environment timesteps at which some agents have stepped.
    See for example `MultiAgentEpisode.get_observations()`.
    """

    def find_indices(self, indices_to_find: List[int]):
        if False:
            i = 10
            return i + 15
        'Returns global timesteps at which an agent stepped.\n\n        The function returns for a given list of indices the ones\n        that are stored in the `IndexMapping`.\n\n        Args:\n            indices_to_find: A list of indices that should be\n                found in the `IndexMapping`.\n\n        Returns:\n            A list of indices at which to find the `indices_to_find`\n            in `self`. This could be empty if none of the given\n            indices are in `IndexMapping`.\n        '
        indices = []
        for num in indices_to_find:
            if num in self:
                indices.append(self.index(num))
        return indices