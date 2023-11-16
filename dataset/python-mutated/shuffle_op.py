"""The implementation of `tf.data.Dataset.shuffle`."""
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops

def _shuffle(input_dataset, buffer_size, seed=None, reshuffle_each_iteration=None, name=None):
    if False:
        return 10
    return _ShuffleDataset(input_dataset, buffer_size, seed, reshuffle_each_iteration, name=name)

class _ShuffleDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that randomly shuffles the elements of its input."""

    def __init__(self, input_dataset, buffer_size, seed=None, reshuffle_each_iteration=None, name=None):
        if False:
            print('Hello World!')
        'See `Dataset.shuffle()` for details.'
        self._input_dataset = input_dataset
        self._buffer_size = ops.convert_to_tensor(buffer_size, dtype=dtypes.int64, name='buffer_size')
        (self._seed, self._seed2) = random_seed.get_seed(seed)
        if reshuffle_each_iteration is None:
            reshuffle_each_iteration = True
        self._reshuffle_each_iteration = reshuffle_each_iteration
        self._name = name
        if tf2.enabled() and (context.executing_eagerly() or ops.inside_function()):
            variant_tensor = gen_dataset_ops.shuffle_dataset_v3(input_dataset._variant_tensor, buffer_size=self._buffer_size, seed=self._seed, seed2=self._seed2, seed_generator=gen_dataset_ops.dummy_seed_generator(), reshuffle_each_iteration=self._reshuffle_each_iteration, **self._common_args)
        else:
            variant_tensor = gen_dataset_ops.shuffle_dataset(input_dataset._variant_tensor, buffer_size=self._buffer_size, seed=self._seed, seed2=self._seed2, reshuffle_each_iteration=self._reshuffle_each_iteration, **self._common_args)
        super().__init__(input_dataset, variant_tensor)