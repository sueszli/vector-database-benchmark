from collections import deque
import copy
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.base import ReplayBufferInterface
from ray.rllib.utils.typing import SampleBatchType

class EpisodeReplayBuffer(ReplayBufferInterface):
    """Buffer that stores (completed or truncated) episodes by their ID.

    Each "row" (a slot in a deque) in the buffer is occupied by one episode. If an
    incomplete episode is added to the buffer and then another chunk of that episode is
    added at a later time, the buffer will automatically concatenate the new fragment to
    the original episode. This way, episodes can be completed via subsequent `add`
    calls.

    Sampling returns batches of size B (number of "rows"), where each row is a
    trajectory of length T. Each trajectory contains consecutive timesteps from an
    episode, but might not start at the beginning of that episode. Should an episode end
    within such a trajectory, a random next episode (starting from its t0) will be
    concatenated to that "row". Example: `sample(B=2, T=4)` ->

       0 .. 1 .. 2 .. 3  <- T-axis
    0 e5   e6   e7   e8
    1 f2   f3   h0   h2
    ^ B-axis

    .. where e, f, and h are different (randomly picked) episodes, the 0-index (e.g. h0)
    indicates the start of an episode, and `f3` is an episode end (gym environment
    returned terminated=True or truncated=True).

    0-indexed returned timesteps contain the reset observation, a dummy 0.0 reward, as
    well as the first action taken in the episode (action picked after observing
    obs(0)).
    The last index in an episode (e.g. f3 in the example above) contains the final
    observation of the episode, the final reward received, a dummy action
    (repeat the previous action), as well as either terminated=True or truncated=True.
    """

    def __init__(self, capacity: int=10000, *, batch_size_B: int=16, batch_length_T: int=64):
        if False:
            for i in range(10):
                print('nop')
        'Initializes an EpisodeReplayBuffer instance.\n\n        Args:\n            capacity: The total number of timesteps to be storable in this buffer.\n                Will start ejecting old episodes once this limit is reached.\n            batch_size_B: The number of rows in a SampleBatch returned from `sample()`.\n            batch_length_T: The length of each row in a SampleBatch returned from\n                `sample()`.\n        '
        self.capacity = capacity
        self.batch_size_B = batch_size_B
        self.batch_length_T = batch_length_T
        self.episodes = deque()
        self.episode_id_to_index = {}
        self._num_episodes_evicted = 0
        self._indices = []
        self._num_timesteps = 0
        self._num_timesteps_added = 0
        self.sampled_timesteps = 0
        self.rng = np.random.default_rng(seed=None)

    @override(ReplayBufferInterface)
    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return self.get_num_timesteps()

    @override(ReplayBufferInterface)
    def add(self, episodes: Union[List['SingleAgentEpisode'], 'SingleAgentEpisode']):
        if False:
            return 10
        'Converts the incoming SampleBatch into a number of SingleAgentEpisode objects.\n\n        Then adds these episodes to the internal deque.\n        '
        if isinstance(episodes, SingleAgentEpisode):
            episodes = [episodes]
        for eps in episodes:
            eps = copy.deepcopy(eps)
            self._num_timesteps += len(eps)
            self._num_timesteps_added += len(eps)
            if eps.id_ in self.episode_id_to_index:
                eps_idx = self.episode_id_to_index[eps.id_]
                existing_eps = self.episodes[eps_idx - self._num_episodes_evicted]
                old_len = len(existing_eps)
                self._indices.extend([(eps_idx, old_len + i) for i in range(len(eps))])
                existing_eps.concat_episode(eps)
            else:
                self.episodes.append(eps)
                eps_idx = len(self.episodes) - 1 + self._num_episodes_evicted
                self.episode_id_to_index[eps.id_] = eps_idx
                self._indices.extend([(eps_idx, i) for i in range(len(eps))])
            while self._num_timesteps > self.capacity and self.get_num_episodes() > 1:
                evicted_eps = self.episodes.popleft()
                evicted_eps_len = len(evicted_eps)
                self._num_timesteps -= evicted_eps_len
                evicted_idx = self.episode_id_to_index[evicted_eps.id_]
                del self.episode_id_to_index[evicted_eps.id_]
                new_indices = []
                idx_cursor = 0
                for (i, idx_tuple) in enumerate(self._indices):
                    if idx_cursor is not None and idx_tuple[0] == evicted_idx:
                        new_indices.extend(self._indices[idx_cursor:i])
                        idx_cursor = None
                    elif idx_cursor is None:
                        if idx_tuple[0] != evicted_idx:
                            idx_cursor = i
                            if evicted_eps_len == 1:
                                break
                        elif idx_tuple[1] == evicted_eps_len - 1:
                            assert self._indices[i + 1][0] != idx_tuple[0]
                            idx_cursor = i + 1
                            break
                if idx_cursor is not None:
                    new_indices.extend(self._indices[idx_cursor:])
                self._indices = new_indices
                self._num_episodes_evicted += 1

    @override(ReplayBufferInterface)
    def sample(self, num_items: Optional[int]=None, *, batch_size_B: Optional[int]=None, batch_length_T: Optional[int]=None) -> SampleBatchType:
        if False:
            while True:
                i = 10
        'Returns a batch of size B (number of "rows"), where each row has length T.\n\n        Each row contains consecutive timesteps from an episode, but might not start\n        at the beginning of that episode. Should an episode end within such a\n        row (trajectory), a random next episode (starting from its t0) will be\n        concatenated to that row. For more details, see the docstring of the\n        EpisodeReplayBuffer class.\n\n        Args:\n            num_items: See `batch_size_B`. For compatibility with the\n                `ReplayBufferInterface` abstract base class.\n            batch_size_B: The number of rows (trajectories) to return in the batch.\n            batch_length_T: The length of each row (in timesteps) to return in the\n                batch.\n\n        Returns:\n            The sampled batch (observations, actions, rewards, terminateds, truncateds)\n                of dimensions [B, T, ...].\n        '
        if num_items is not None:
            assert batch_size_B is None, 'Cannot call `sample()` with both `num_items` and `batch_size_B` provided! Use either one.'
            batch_size_B = num_items
        batch_size_B = batch_size_B or self.batch_size_B
        batch_length_T = batch_length_T or self.batch_length_T
        observations = [[] for _ in range(batch_size_B)]
        actions = [[] for _ in range(batch_size_B)]
        rewards = [[] for _ in range(batch_size_B)]
        is_first = [[False] * batch_length_T for _ in range(batch_size_B)]
        is_last = [[False] * batch_length_T for _ in range(batch_size_B)]
        is_terminated = [[False] * batch_length_T for _ in range(batch_size_B)]
        is_truncated = [[False] * batch_length_T for _ in range(batch_size_B)]
        B = 0
        T = 0
        while B < batch_size_B:
            index_tuple = self._indices[self.rng.integers(len(self._indices))]
            (episode_idx, episode_ts) = (index_tuple[0] - self._num_episodes_evicted, index_tuple[1])
            episode = self.episodes[episode_idx]
            is_first[B][T] = True
            if len(rewards[B]) == 0:
                if episode_ts == 0:
                    rewards[B].append(0.0)
                else:
                    rewards[B].append(episode.rewards[episode_ts - 1])
            else:
                episode_ts = 0
                rewards[B].append(0.0)
            observations[B].extend(episode.observations[episode_ts:])
            actions[B].extend(episode.actions[episode_ts:])
            actions[B].append(episode.actions[-1])
            rewards[B].extend(episode.rewards[episode_ts:])
            assert len(observations[B]) == len(actions[B]) == len(rewards[B])
            T = min(len(observations[B]), batch_length_T)
            is_last[B][T - 1] = True
            if episode.is_terminated and T == len(observations[B]):
                is_terminated[B][T - 1] = True
            elif episode.is_truncated and T == len(observations[B]):
                is_truncated[B][T - 1] = True
            if T == batch_length_T:
                observations[B] = observations[B][:batch_length_T]
                actions[B] = actions[B][:batch_length_T]
                rewards[B] = rewards[B][:batch_length_T]
                B += 1
                T = 0
        self.sampled_timesteps += batch_size_B * batch_length_T
        ret = {'obs': np.array(observations), 'actions': np.array(actions), 'rewards': np.array(rewards), 'is_first': np.array(is_first), 'is_last': np.array(is_last), 'is_terminated': np.array(is_terminated), 'is_truncated': np.array(is_truncated)}
        return ret

    def get_num_episodes(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns number of episodes (completed or truncated) stored in the buffer.'
        return len(self.episodes)

    def get_num_timesteps(self) -> int:
        if False:
            i = 10
            return i + 15
        'Returns number of individual timesteps stored in the buffer.'
        return len(self._indices)

    def get_sampled_timesteps(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        "Returns number of timesteps that have been sampled in buffer's lifetime."
        return self.sampled_timesteps

    def get_added_timesteps(self) -> int:
        if False:
            i = 10
            return i + 15
        "Returns number of timesteps that have been added in buffer's lifetime."
        return self._num_timesteps_added

    @override(ReplayBufferInterface)
    def get_state(self) -> Dict[str, Any]:
        if False:
            return 10
        return {'episodes': [eps.get_state() for eps in self.episodes], 'episode_id_to_index': list(self.episode_id_to_index.items()), '_num_episodes_evicted': self._num_episodes_evicted, '_indices': self._indices, '_num_timesteps': self._num_timesteps, '_num_timesteps_added': self._num_timesteps_added, 'sampled_timesteps': self.sampled_timesteps}

    @override(ReplayBufferInterface)
    def set_state(self, state) -> None:
        if False:
            i = 10
            return i + 15
        self.episodes = deque([SingleAgentEpisode.from_state(eps_data) for eps_data in state['episodes']])
        self.episode_id_to_index = dict(state['episode_id_to_index'])
        self._num_episodes_evicted = state['_num_episodes_evicted']
        self._indices = state['_indices']
        self._num_timesteps = state['_num_timesteps']
        self._num_timesteps_added = state['_num_timesteps_added']
        self.sampled_timesteps = state['sampled_timesteps']