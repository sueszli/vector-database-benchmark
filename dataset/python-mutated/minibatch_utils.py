import math
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch

@DeveloperAPI
class MiniBatchIteratorBase:
    """The base class for all minibatch iterators.

    Args:
        batch: The input multi-agent batch.
        minibatch_size: The size of the minibatch for each module_id.
        num_iters: The number of epochs to cover. If the input batch is smaller than
            minibatch_size, then the iterator will cycle through the batch until it
            has covered num_iters epochs.
    """

    def __init__(self, batch: MultiAgentBatch, minibatch_size: int, num_iters: int=1) -> None:
        if False:
            print('Hello World!')
        pass

@DeveloperAPI
class MiniBatchCyclicIterator(MiniBatchIteratorBase):
    """This implements a simple multi-agent minibatch iterator.


    This iterator will split the input multi-agent batch into minibatches where the
    size of batch for each module_id (aka policy_id) is equal to minibatch_size. If the
    input batch is smaller than minibatch_size, then the iterator will cycle through
    the batch until it has covered num_iters epochs.

    Args:
        batch: The input multi-agent batch.
        minibatch_size: The size of the minibatch for each module_id.
        num_iters: The minimum number of epochs to cover. If the input batch is smaller
            than minibatch_size, then the iterator will cycle through the batch until
            it has covered at least num_iters epochs.
    """

    def __init__(self, batch: MultiAgentBatch, minibatch_size: int, num_iters: int=1) -> None:
        if False:
            return 10
        super().__init__(batch, minibatch_size, num_iters)
        self._batch = batch
        self._minibatch_size = minibatch_size
        self._num_iters = num_iters
        self._start = {mid: 0 for mid in batch.policy_batches.keys()}
        self._num_covered_epochs = {mid: 0 for mid in batch.policy_batches.keys()}

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        while min(self._num_covered_epochs.values()) < self._num_iters:
            minibatch = {}
            for (module_id, module_batch) in self._batch.policy_batches.items():
                if len(module_batch) == 0:
                    raise ValueError(f'The batch for module_id {module_id} is empty! This will create an infinite loop because we need to cover the same number of samples for each module_id.')
                s = self._start[module_id]
                n_steps = self._minibatch_size
                samples_to_concat = []
                if module_batch._slice_seq_lens_in_B:
                    assert module_batch.get(SampleBatch.SEQ_LENS) is not None, 'MiniBatchCyclicIterator requires SampleBatch.SEQ_LENSto be present in the batch for slicing a batch in the batch dimension B.'

                    def get_len(b):
                        if False:
                            i = 10
                            return i + 15
                        return len(b[SampleBatch.SEQ_LENS])
                else:

                    def get_len(b):
                        if False:
                            for i in range(10):
                                print('nop')
                        return len(b)
                while n_steps >= get_len(module_batch) - s:
                    sample = module_batch[s:]
                    samples_to_concat.append(sample)
                    len_sample = get_len(sample)
                    assert len_sample > 0, 'Length of a sample must be > 0!'
                    n_steps -= len_sample
                    s = 0
                    self._num_covered_epochs[module_id] += 1
                e = s + n_steps
                if e > s:
                    samples_to_concat.append(module_batch[s:e])
                minibatch[module_id] = concat_samples(samples_to_concat)
                self._start[module_id] = e
            minibatch = MultiAgentBatch(minibatch, len(self._batch))
            yield minibatch

class MiniBatchDummyIterator(MiniBatchIteratorBase):

    def __init__(self, batch: MultiAgentBatch, minibatch_size: int, num_iters: int=1):
        if False:
            return 10
        super().__init__(batch, minibatch_size, num_iters)
        self._batch = batch

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        yield self._batch

@DeveloperAPI
class ShardBatchIterator:
    """Iterator for sharding batch into num_shards batches.

    Args:
        batch: The input multi-agent batch.
        num_shards: The number of shards to split the batch into.

    Yields:
        A MultiAgentBatch of size len(batch) / num_shards.
    """

    def __init__(self, batch: MultiAgentBatch, num_shards: int):
        if False:
            while True:
                i = 10
        self._batch = batch
        self._num_shards = num_shards

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(self._num_shards):
            batch_to_send = {}
            for (pid, sub_batch) in self._batch.policy_batches.items():
                batch_size = math.ceil(len(sub_batch) / self._num_shards)
                start = batch_size * i
                end = min(start + batch_size, len(sub_batch))
                batch_to_send[pid] = sub_batch[int(start):int(end)]
            new_batch = MultiAgentBatch(batch_to_send, int(batch_size))
            yield new_batch