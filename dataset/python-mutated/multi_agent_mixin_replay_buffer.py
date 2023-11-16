import collections
import logging
import random
from typing import Any, Dict, Optional
import numpy as np
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch, concat_samples_into_ma_batch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer, ReplayMode, merge_dicts_with_warning
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES, StorageUnit
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
logger = logging.getLogger(__name__)

@DeveloperAPI
class MultiAgentMixInReplayBuffer(MultiAgentPrioritizedReplayBuffer):
    """This buffer adds replayed samples to a stream of new experiences.

    - Any newly added batch (`add()`) is immediately returned upon
    the next `sample` call (close to on-policy) as well as being moved
    into the buffer.
    - Additionally, a certain number of old samples is mixed into the
    returned sample according to a given "replay ratio".
    - If >1 calls to `add()` are made without any `sample()` calls
    in between, all newly added batches are returned (plus some older samples
    according to the "replay ratio").

    .. testcode::
        :skipif: True

        # replay ratio 0.66 (2/3 replayed, 1/3 new samples):
        buffer = MultiAgentMixInReplayBuffer(capacity=100,
                                             replay_ratio=0.66)
        buffer.add(<A>)
        buffer.add(<B>)
        buffer.sample(1)

    .. testoutput::

        ..[<A>, <B>, <B>]

    .. testcode::
        :skipif: True

        buffer.add(<C>)
        buffer.sample(1)

    .. testoutput::

        [<C>, <A>, <B>]
        or: [<C>, <A>, <A>], [<C>, <B>, <A>] or [<C>, <B>, <B>],
        but always <C> as it is the newest sample

    .. testcode::
        :skipif: True

        buffer.add(<D>)
        buffer.sample(1)

    .. testoutput::

        [<D>, <A>, <C>]
        or [<D>, <A>, <A>], [<D>, <B>, <A>] or [<D>, <B>, <C>], etc..
        but always <D> as it is the newest sample

    .. testcode::
        :skipif: True

        # replay proportion 0.0 -> replay disabled:
        buffer = MixInReplay(capacity=100, replay_ratio=0.0)
        buffer.add(<A>)
        buffer.sample()

    .. testoutput::

        [<A>]

    .. testcode::
        :skipif: True

        buffer.add(<B>)
        buffer.sample()

    .. testoutput::

        [<B>]
    """

    def __init__(self, capacity: int=10000, storage_unit: str='timesteps', num_shards: int=1, replay_mode: str='independent', replay_sequence_override: bool=True, replay_sequence_length: int=1, replay_burn_in: int=0, replay_zero_init_states: bool=True, replay_ratio: float=0.66, underlying_buffer_config: dict=None, prioritized_replay_alpha: float=0.6, prioritized_replay_beta: float=0.4, prioritized_replay_eps: float=1e-06, **kwargs):
        if False:
            print('Hello World!')
        'Initializes MultiAgentMixInReplayBuffer instance.\n\n        Args:\n            capacity: The capacity of the buffer, measured in `storage_unit`.\n            storage_unit: Either \'timesteps\', \'sequences\' or\n                \'episodes\'. Specifies how experiences are stored. If they\n                are stored in episodes, replay_sequence_length is ignored.\n            num_shards: The number of buffer shards that exist in total\n                (including this one).\n            replay_mode: One of "independent" or "lockstep". Determines,\n                whether batches are sampled independently or to an equal\n                amount.\n            replay_sequence_override: If True, ignore sequences found in incoming\n                batches, slicing them into sequences as specified by\n                `replay_sequence_length` and `replay_sequence_burn_in`. This only has\n                an effect if storage_unit is `sequences`.\n            replay_sequence_length: The sequence length (T) of a single\n                sample. If > 1, we will sample B x T from this buffer. This\n                only has an effect if storage_unit is \'timesteps\'.\n            replay_burn_in: The burn-in length in case\n                `replay_sequence_length` > 0. This is the number of timesteps\n                each sequence overlaps with the previous one to generate a\n                better internal state (=state after the burn-in), instead of\n                starting from 0.0 each RNN rollout.\n            replay_zero_init_states: Whether the initial states in the\n                buffer (if replay_sequence_length > 0) are alwayas 0.0 or\n                should be updated with the previous train_batch state outputs.\n            replay_ratio: Ratio of replayed samples in the returned\n                batches. E.g. a ratio of 0.0 means only return new samples\n                (no replay), a ratio of 0.5 means always return newest sample\n                plus one old one (1:1), a ratio of 0.66 means always return\n                the newest sample plus 2 old (replayed) ones (1:2), etc...\n            underlying_buffer_config: A config that contains all necessary\n                constructor arguments and arguments for methods to call on\n                the underlying buffers. This replaces the standard behaviour\n                of the underlying PrioritizedReplayBuffer. The config\n                follows the conventions of the general\n                replay_buffer_config. kwargs for subsequent calls of methods\n                may also be included. Example:\n                "replay_buffer_config": {"type": PrioritizedReplayBuffer,\n                "capacity": 10, "storage_unit": "timesteps",\n                prioritized_replay_alpha: 0.5, prioritized_replay_beta: 0.5,\n                prioritized_replay_eps: 0.5}\n            prioritized_replay_alpha: Alpha parameter for a prioritized\n                replay buffer. Use 0.0 for no prioritization.\n            prioritized_replay_beta: Beta parameter for a prioritized\n                replay buffer.\n            prioritized_replay_eps: Epsilon parameter for a prioritized\n                replay buffer.\n            **kwargs: Forward compatibility kwargs.\n        '
        if not 0 <= replay_ratio <= 1:
            raise ValueError('Replay ratio must be within [0, 1]')
        MultiAgentPrioritizedReplayBuffer.__init__(self, capacity=capacity, storage_unit=storage_unit, num_shards=num_shards, replay_mode=replay_mode, replay_sequence_override=replay_sequence_override, replay_sequence_length=replay_sequence_length, replay_burn_in=replay_burn_in, replay_zero_init_states=replay_zero_init_states, underlying_buffer_config=underlying_buffer_config, prioritized_replay_alpha=prioritized_replay_alpha, prioritized_replay_beta=prioritized_replay_beta, prioritized_replay_eps=prioritized_replay_eps, **kwargs)
        self.replay_ratio = replay_ratio
        self.last_added_batches = collections.defaultdict(list)

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        if False:
            print('Hello World!')
        "Adds a batch to the appropriate policy's replay buffer.\n\n        Turns the batch into a MultiAgentBatch of the DEFAULT_POLICY_ID if\n        it is not a MultiAgentBatch. Subsequently, adds the individual policy\n        batches to the storage.\n\n        Args:\n            batch: The batch to be added.\n            **kwargs: Forward compatibility kwargs.\n        "
        batch = batch.copy()
        batch = batch.as_multi_agent()
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)
        pids_and_batches = self._maybe_split_into_policy_batches(batch)
        with self.add_batch_timer:
            if self.storage_unit == StorageUnit.TIMESTEPS:
                for (policy_id, sample_batch) in pids_and_batches.items():
                    timeslices = sample_batch.timeslices(1)
                    for time_slice in timeslices:
                        self.replay_buffers[policy_id].add(time_slice, **kwargs)
                        self.last_added_batches[policy_id].append(time_slice)
            elif self.storage_unit == StorageUnit.SEQUENCES:
                for (policy_id, sample_batch) in pids_and_batches.items():
                    timeslices = timeslice_along_seq_lens_with_overlap(sample_batch=sample_batch, seq_lens=sample_batch.get(SampleBatch.SEQ_LENS) if self.replay_sequence_override else None, zero_pad_max_seq_len=self.replay_sequence_length, pre_overlap=self.replay_burn_in, zero_init_states=self.replay_zero_init_states)
                    for slice in timeslices:
                        self.replay_buffers[policy_id].add(slice, **kwargs)
                        self.last_added_batches[policy_id].append(slice)
            elif self.storage_unit == StorageUnit.EPISODES:
                for (policy_id, sample_batch) in pids_and_batches.items():
                    for eps in sample_batch.split_by_episode():
                        if eps.get(SampleBatch.T)[0] == 0 and (eps.get(SampleBatch.TERMINATEDS, [True])[-1] or eps.get(SampleBatch.TRUNCATEDS, [False])[-1]):
                            self.replay_buffers[policy_id].add(eps, **kwargs)
                            self.last_added_batches[policy_id].append(eps)
                        elif log_once('only_full_episodes'):
                            logger.info('This buffer uses episodes as a storage unit and thus allows only full episodes to be added to it. Some samples may be dropped.')
            elif self.storage_unit == StorageUnit.FRAGMENTS:
                for (policy_id, sample_batch) in pids_and_batches.items():
                    self.replay_buffers[policy_id].add(sample_batch, **kwargs)
                    self.last_added_batches[policy_id].append(sample_batch)
        self._num_added += batch.count

    @DeveloperAPI
    @override(MultiAgentReplayBuffer)
    def sample(self, num_items: int, policy_id: PolicyID=DEFAULT_POLICY_ID, **kwargs) -> Optional[SampleBatchType]:
        if False:
            for i in range(10):
                print('nop')
        'Samples a batch of size `num_items` from a specified buffer.\n\n        Concatenates old samples to new ones according to\n        self.replay_ratio. If not enough new samples are available, mixes in\n        less old samples to retain self.replay_ratio on average. Returns\n        an empty batch if there are no items in the buffer.\n\n        Args:\n            num_items: Number of items to sample from this buffer.\n            policy_id: ID of the policy that produced the experiences to be\n            sampled.\n            **kwargs: Forward compatibility kwargs.\n\n        Returns:\n            Concatenated MultiAgentBatch of items.\n        '
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        def mix_batches(_policy_id):
            if False:
                for i in range(10):
                    print('nop')
            'Mixes old with new samples.\n\n            Tries to mix according to self.replay_ratio on average.\n            If not enough new samples are available, mixes in less old samples\n            to retain self.replay_ratio on average.\n            '

            def round_up_or_down(value, ratio):
                if False:
                    print('Hello World!')
                'Returns an integer averaging to value*ratio.'
                product = value * ratio
                ceil_prob = product % 1
                if random.uniform(0, 1) < ceil_prob:
                    return int(np.ceil(product))
                else:
                    return int(np.floor(product))
            max_num_new = round_up_or_down(num_items, 1 - self.replay_ratio)
            _buffer = self.replay_buffers[_policy_id]
            output_batches = self.last_added_batches[_policy_id][:max_num_new]
            self.last_added_batches[_policy_id] = self.last_added_batches[_policy_id][max_num_new:]
            if self.replay_ratio == 0.0:
                return concat_samples_into_ma_batch(output_batches)
            elif self.replay_ratio == 1.0:
                return _buffer.sample(num_items, **kwargs)
            num_new = len(output_batches)
            if np.isclose(num_new, num_items * (1 - self.replay_ratio)):
                num_old = num_items - max_num_new
            else:
                num_old = min(num_items - max_num_new, round_up_or_down(num_new, self.replay_ratio / (1 - self.replay_ratio)))
            output_batches.append(_buffer.sample(num_old, **kwargs))
            output_batches = [batch.as_multi_agent() for batch in output_batches]
            return concat_samples_into_ma_batch(output_batches)

        def check_buffer_is_ready(_policy_id):
            if False:
                print('Hello World!')
            if len(self.replay_buffers[policy_id]) == 0 and self.replay_ratio > 0.0 or (len(self.last_added_batches[_policy_id]) == 0 and self.replay_ratio < 1.0):
                return False
            return True
        with self.replay_timer:
            samples = []
            if self.replay_mode == ReplayMode.LOCKSTEP:
                assert policy_id is None, '`policy_id` specifier not allowed in `lockstep` mode!'
                if check_buffer_is_ready(_ALL_POLICIES):
                    samples.append(mix_batches(_ALL_POLICIES).as_multi_agent())
            elif policy_id is not None:
                if check_buffer_is_ready(policy_id):
                    samples.append(mix_batches(policy_id).as_multi_agent())
            else:
                for (policy_id, replay_buffer) in self.replay_buffers.items():
                    if check_buffer_is_ready(policy_id):
                        samples.append(mix_batches(policy_id).as_multi_agent())
            return concat_samples_into_ma_batch(samples)

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def get_state(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Returns all local state.\n\n        Returns:\n            The serializable local state.\n        '
        data = {'last_added_batches': self.last_added_batches}
        parent = MultiAgentPrioritizedReplayBuffer.get_state(self)
        parent.update(data)
        return parent

    @DeveloperAPI
    @override(MultiAgentPrioritizedReplayBuffer)
    def set_state(self, state: Dict[str, Any]) -> None:
        if False:
            return 10
        'Restores all local state to the provided `state`.\n\n        Args:\n            state: The new state to set this buffer. Can be obtained by\n                calling `self.get_state()`.\n        '
        self.last_added_batches = state['last_added_batches']
        MultiAgentPrioritizedReplayBuffer.set_state(state)