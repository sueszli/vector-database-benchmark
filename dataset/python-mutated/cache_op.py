"""The implementation of `tf.data.Dataset.cache`."""
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops

def _cache(input_dataset, filename, name):
    if False:
        i = 10
        return i + 15
    return CacheDataset(input_dataset, filename, name)

class CacheDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that caches elements of its input."""

    def __init__(self, input_dataset, filename, name=None):
        if False:
            i = 10
            return i + 15
        'See `Dataset.cache()` for details.'
        self._input_dataset = input_dataset
        self._filename = ops.convert_to_tensor(filename, dtype=dtypes.string, name='filename')
        self._name = name
        if tf2.enabled() and (context.executing_eagerly() or ops.inside_function()):
            variant_tensor = gen_dataset_ops.cache_dataset_v2(input_dataset._variant_tensor, filename=self._filename, cache=gen_dataset_ops.dummy_memory_cache(), **self._common_args)
        else:
            variant_tensor = gen_dataset_ops.cache_dataset(input_dataset._variant_tensor, filename=self._filename, **self._common_args)
        super().__init__(input_dataset, variant_tensor)