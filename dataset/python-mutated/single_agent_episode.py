import numpy as np
import uuid
from gymnasium.core import ActType, ObsType
from typing import Any, Dict, List, Optional, SupportsFloat
from ray.rllib.policy.sample_batch import SampleBatch

class SingleAgentEpisode:

    def __init__(self, id_: Optional[str]=None, *, observations: List[ObsType]=None, actions: List[ActType]=None, rewards: List[SupportsFloat]=None, infos: List[Dict]=None, states=None, t_started: Optional[int]=None, is_terminated: bool=False, is_truncated: bool=False, render_images: Optional[List[np.ndarray]]=None, extra_model_outputs: Optional[Dict[str, Any]]=None) -> 'SingleAgentEpisode':
        if False:
            for i in range(10):
                print('nop')
        'Initializes a `SingleAgentEpisode` instance.\n\n        This constructor can be called with or without sampled data. Note\n        that if data is provided the episode will start at timestep\n        `t_started = len(observations) - 1` (the initial observation is not\n        counted). If the episode should start at `t_started = 0` (e.g.\n        because the instance should simply store episode data) this has to\n        be provided in the `t_started` parameter of the constructor.\n\n        Args:\n            id_: Optional. Unique identifier for this episode. If no id is\n                provided the constructor generates a hexadecimal code for the id.\n            observations: Optional. A list of observations from a rollout. If\n                data is provided it should be complete (i.e. observations, actions,\n                rewards, is_terminated, is_truncated, and all necessary\n                `extra_model_outputs`). The length of the `observations` defines\n                the default starting value. See the parameter `t_started`.\n            actions: Optional. A list of actions from a rollout. If data is\n                provided it should be complete (i.e. observations, actions,\n                rewards, is_terminated, is_truncated, and all necessary\n                `extra_model_outputs`).\n            rewards: Optional. A list of rewards from a rollout. If data is\n                provided it should be complete (i.e. observations, actions,\n                rewards, is_terminated, is_truncated, and all necessary\n                `extra_model_outputs`).\n            infos: Optional. A list of infos from a rollout. If data is\n                provided it should be complete (i.e. observations, actions,\n                rewards, is_terminated, is_truncated, and all necessary\n                `extra_model_outputs`).\n            states: Optional. The hidden model states from a rollout. If\n                data is provided it should be complete (i.e. observations, actions,\n                rewards, is_terminated, is_truncated, and all necessary\n                `extra_model_outputs`). States are only avasilable if a stateful\n                model (`RLModule`) is used.\n            t_started: Optional. The starting timestep of the episode. The default\n                is zero. If data is provided, the starting point is from the last\n                observation onwards (i.e. `t_started = len(observations) - 1). If\n                this parameter is provided the episode starts at the provided value.\n            is_terminated: Optional. A boolean indicating, if the episode is already\n                terminated. Note, this parameter is only needed, if episode data is\n                provided in the constructor. The default is `False`.\n            is_truncated: Optional. A boolean indicating, if the episode was\n                truncated. Note, this parameter is only needed, if episode data is\n                provided in the constructor. The default is `False`.\n            render_images: Optional. A list of RGB uint8 images from rendering\n                the environment.\n            extra_model_outputs: Optional. A list of dictionaries containing specific\n                model outputs for the algorithm used (e.g. `vf_preds` and `action_logp`\n                for PPO) from a rollout. If data is provided it should be complete\n                (i.e. observations, actions, rewards, is_terminated, is_truncated,\n                and all necessary `extra_model_outputs`).\n        '
        self.id_ = id_ or uuid.uuid4().hex
        self.observations = [] if observations is None else observations
        self.actions = [] if actions is None else actions
        self.rewards = [] if rewards is None else rewards
        if infos is None:
            self.infos = [{} for _ in range(len(self.observations))]
        else:
            self.infos = infos
        self.states = states
        self.t = self.t_started = t_started if t_started is not None else max(len(self.observations) - 1, 0)
        if self.t_started < len(self.observations) - 1:
            self.t = len(self.observations) - 1
        self.is_terminated = is_terminated
        self.is_truncated = is_truncated
        assert render_images is None or observations is not None
        self.render_images = [] if render_images is None else render_images
        self.extra_model_outputs = {} if extra_model_outputs is None else extra_model_outputs

    def concat_episode(self, episode_chunk: 'SingleAgentEpisode'):
        if False:
            return 10
        'Adds the given `episode_chunk` to the right side of self.\n\n        Args:\n            episode_chunk: Another `SingleAgentEpisode` to be concatenated.\n\n        Returns: A `SingleAegntEpisode` instance containing the concatenated\n            from both episodes.\n        '
        assert episode_chunk.id_ == self.id_
        assert not self.is_done
        assert self.t == episode_chunk.t_started
        episode_chunk.validate()
        assert np.all(episode_chunk.observations[0] == self.observations[-1])
        self.observations.pop()
        self.infos.pop()
        self.observations.extend(list(episode_chunk.observations))
        self.actions.extend(list(episode_chunk.actions))
        self.rewards.extend(list(episode_chunk.rewards))
        self.infos.extend(list(episode_chunk.infos))
        self.t = episode_chunk.t
        self.states = episode_chunk.states
        if episode_chunk.is_terminated:
            self.is_terminated = True
        elif episode_chunk.is_truncated:
            self.is_truncated = True
        for (k, v) in episode_chunk.extra_model_outputs.items():
            self.extra_model_outputs[k].extend(list(v))
        self.validate()

    def add_initial_observation(self, *, initial_observation: ObsType, initial_info: Optional[Dict]=None, initial_state=None, initial_render_image: Optional[np.ndarray]=None) -> None:
        if False:
            return 10
        'Adds the initial data to the episode.\n\n        Args:\n            initial_observation: Obligatory. The initial observation.\n            initial_info: Optional. The initial info.\n            initial_state: Optional. The initial hidden state of a\n                model (`RLModule`) if the latter is stateful.\n            initial_render_image: Optional. An RGB uint8 image from rendering\n                the environment.\n        '
        assert not self.is_done
        assert len(self.observations) == 0
        assert self.t == self.t_started == 0
        initial_info = initial_info or {}
        self.observations.append(initial_observation)
        self.states = initial_state
        self.infos.append(initial_info)
        if initial_render_image is not None:
            self.render_images.append(initial_render_image)
        self.validate()

    def add_timestep(self, observation: ObsType, action: ActType, reward: SupportsFloat, *, info: Optional[Dict[str, Any]]=None, state=None, is_terminated: bool=False, is_truncated: bool=False, render_image: Optional[np.ndarray]=None, extra_model_output: Optional[Dict[str, Any]]=None) -> None:
        if False:
            while True:
                i = 10
        "Adds a timestep to the episode.\n\n        Args:\n            observation: The observation received from the\n                environment.\n            action: The last action used by the agent.\n            reward: The last reward received by the agent.\n            info: The last info recevied from the environment.\n            state: Optional. The last hidden state of the model (`RLModule` ).\n                This is only available, if the model is stateful.\n            is_terminated: A boolean indicating, if the environment has been\n                terminated.\n            is_truncated: A boolean indicating, if the environment has been\n                truncated.\n            render_image: Optional. An RGB uint8 image from rendering\n                the environment.\n            extra_model_output: The last timestep's specific model outputs\n                (e.g. `vf_preds`  for PPO).\n        "
        assert not self.is_done
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        info = info or {}
        self.infos.append(info)
        self.states = state
        self.t += 1
        if render_image is not None:
            self.render_images.append(render_image)
        if extra_model_output is not None:
            for (k, v) in extra_model_output.items():
                if k not in self.extra_model_outputs:
                    self.extra_model_outputs[k] = [v]
                else:
                    self.extra_model_outputs[k].append(v)
        self.is_terminated = is_terminated
        self.is_truncated = is_truncated
        self.validate()

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates the episode.\n\n        This function ensures that the data stored to a `SingleAgentEpisode` is\n        in order (e.g. that the correct number of observations, actions, rewards\n        are there).\n        '
        assert len(self.observations) == len(self.infos) == len(self.rewards) + 1 == len(self.actions) + 1
        assert len(self.rewards) == self.t - self.t_started
        if len(self.extra_model_outputs) > 0:
            for (k, v) in self.extra_model_outputs.items():
                assert len(v) == len(self.observations) - 1
        if self.is_done:
            self.convert_lists_to_numpy()

    @property
    def is_done(self) -> bool:
        if False:
            while True:
                i = 10
        'Whether the episode is actually done (terminated or truncated).\n\n        A done episode cannot be continued via `self.add_timestep()` or being\n        concatenated on its right-side with another episode chunk or being\n        succeeded via `self.create_successor()`.\n        '
        return self.is_terminated or self.is_truncated

    def convert_lists_to_numpy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Converts list attributes to numpy arrays.\n\n        When an episode is terminated or truncated (`self.is_done`) the data\n        will be not anymore touched and instead converted to numpy for later\n        use in postprocessing. This function converts all the data stored\n        into numpy arrays.\n        '
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.infos = np.array(self.infos)
        self.render_images = np.array(self.render_images, dtype=np.uint8)
        for (k, v) in self.extra_model_outputs.items():
            self.extra_model_outputs[k] = np.array(v)

    def create_successor(self) -> 'SingleAgentEpisode':
        if False:
            while True:
                i = 10
        'Returns a successor episode chunk (of len=0) continuing with this one.\n\n        The successor will have the same ID and state as self and its only observation\n        will be the last observation in self. Its length will therefore be 0 (no\n        steps taken yet).\n\n        This method is useful if you would like to discontinue building an episode\n        chunk (b/c you have to return it from somewhere), but would like to have a new\n        episode (chunk) instance to continue building the actual env episode at a later\n        time.\n\n        Returns:\n            The successor Episode chunk of this one with the same ID and state and the\n            only observation being the last observation in self.\n        '
        assert not self.is_done
        return SingleAgentEpisode(id_=self.id_, observations=[self.observations[-1]], infos=[self.infos[-1]], states=self.states, t_started=self.t)

    def to_sample_batch(self) -> SampleBatch:
        if False:
            i = 10
            return i + 15
        "Converts a `SingleAgentEpisode` into a `SampleBatch`.\n\n        Note that `RLlib` is relying in training on the `SampleBatch`  class and\n        therefore episodes have to be converted to this format before training can\n        start.\n\n        Returns:\n            An `ray.rLlib.policy.sample_batch.SampleBatch` instance containing this\n            episode's data.\n        "
        return SampleBatch({SampleBatch.EPS_ID: np.array([self.id_] * len(self)), SampleBatch.OBS: self.observations[:-1], SampleBatch.NEXT_OBS: self.observations[1:], SampleBatch.ACTIONS: self.actions, SampleBatch.REWARDS: self.rewards, SampleBatch.T: list(range(self.t_started, self.t)), SampleBatch.TERMINATEDS: np.array([False] * (len(self) - 1) + [self.is_terminated]), SampleBatch.TRUNCATEDS: np.array([False] * (len(self) - 1) + [self.is_truncated]), SampleBatch.INFOS: self.infos[1:], **self.extra_model_outputs})

    @staticmethod
    def from_sample_batch(batch: SampleBatch) -> 'SingleAgentEpisode':
        if False:
            i = 10
            return i + 15
        "Converts a `SampleBatch` instance into a `SingleAegntEpisode`.\n\n        The `ray.rllib.policy.sample_batch.SampleBatch` class is used in `RLlib`\n        for training an agent's modules (`RLModule`), converting from or to\n        `SampleBatch` can be performed by this function and its counterpart\n        `to_sample_batch()`.\n\n        Args:\n            batch: A `SampleBatch` instance. It should contain only a single episode.\n\n        Returns:\n            An `SingleAegntEpisode` instance containing the data from `batch`.\n        "
        is_done = batch[SampleBatch.TERMINATEDS][-1] or batch[SampleBatch.TRUNCATEDS][-1]
        observations = np.concatenate([batch[SampleBatch.OBS], batch[SampleBatch.NEXT_OBS][None, -1]])
        actions = batch[SampleBatch.ACTIONS]
        rewards = batch[SampleBatch.REWARDS]
        infos = batch[SampleBatch.INFOS]
        infos = np.concatenate([np.array([{}]), infos])
        extra_model_output_keys = []
        for k in batch.keys():
            if k not in [SampleBatch.EPS_ID, SampleBatch.AGENT_INDEX, SampleBatch.ENV_ID, SampleBatch.AGENT_INDEX, SampleBatch.T, SampleBatch.SEQ_LENS, SampleBatch.OBS, SampleBatch.INFOS, SampleBatch.NEXT_OBS, SampleBatch.ACTIONS, SampleBatch.PREV_ACTIONS, SampleBatch.REWARDS, SampleBatch.PREV_REWARDS, SampleBatch.TERMINATEDS, SampleBatch.TRUNCATEDS, SampleBatch.UNROLL_ID, SampleBatch.DONES, SampleBatch.CUR_OBS]:
                extra_model_output_keys.append(k)
        return SingleAgentEpisode(id_=batch[SampleBatch.EPS_ID][0], observations=observations if is_done else observations.tolist(), actions=actions if is_done else actions.tolist(), rewards=rewards if is_done else rewards.tolist(), t_started=batch[SampleBatch.T][0], is_terminated=batch[SampleBatch.TERMINATEDS][-1], is_truncated=batch[SampleBatch.TRUNCATEDS][-1], infos=infos if is_done else infos.tolist(), extra_model_outputs={k: batch[k] if is_done else batch[k].tolist() for k in extra_model_output_keys})

    def get_return(self) -> float:
        if False:
            i = 10
            return i + 15
        "Calculates an episode's return.\n\n        The return is computed by a simple sum, neglecting the discount factor.\n        This is used predominantly for metrics.\n\n        Returns:\n            The sum of rewards collected during this episode.\n        "
        return sum(self.rewards)

    def get_state(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Returns the pickable state of an episode.\n\n        The data in the episode is stored into a dictionary. Note that episodes\n        can also be generated from states (see `self.from_state()`).\n\n        Returns:\n            A dictionary containing all the data from the episode.\n        '
        return list({'id_': self.id_, 'observations': self.observations, 'actions': self.actions, 'rewards': self.rewards, 'infos': self.infos, 'states': self.states, 't_started': self.t_started, 't': self.t, 'is_terminated': self.is_terminated, 'is_truncated': self.is_truncated, **self.extra_model_outputs}.items())

    @staticmethod
    def from_state(state: Dict[str, Any]) -> 'SingleAgentEpisode':
        if False:
            print('Hello World!')
        'Generates a `SingleAegntEpisode` from a pickable state.\n\n        The data in the state has to be complete. This is always the case when the state\n        was created by a `SingleAgentEpisode` itself calling `self.get_state()`.\n\n        Args:\n            state: A dictionary containing all episode data.\n\n        Returns:\n            A `SingleAgentEpisode` instance holding all the data provided by `state`.\n        '
        eps = SingleAgentEpisode(id_=state[0][1])
        eps.observations = state[1][1]
        eps.actions = state[2][1]
        eps.rewards = state[3][1]
        eps.infos = state[4][1]
        eps.states = state[5][1]
        eps.t_started = state[6][1]
        eps.t = state[7][1]
        eps.is_terminated = state[8][1]
        eps.is_truncated = state[9][1]
        eps.extra_model_outputs = {k: v for (k, v) in state[10:]}
        eps.validate()
        return eps

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returning the length of an episode.\n\n        The length of an episode is defined by the length of its data. This is the\n        number of timesteps an agent has stepped through an environment so far.\n        The length is undefined in case of a just started episode.\n\n        Returns:\n            An integer, defining the length of an episode.\n\n        Raises:\n            AssertionError: If episode has never been stepped so far.\n        '
        assert len(self.observations) > 0, "ERROR: Cannot determine length of episode that hasn't started yet! Call `SingleAgentEpisode.add_initial_observation(initial_observation=...)` first (after which `len(SingleAgentEpisode)` will be 0)."
        return len(self.observations) - 1