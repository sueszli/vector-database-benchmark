"""The implementation of `tf.data.Dataset.unique`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_experimental_dataset_ops

def _unique(input_dataset, name):
    if False:
        i = 10
        return i + 15
    return _UniqueDataset(input_dataset, name)

class _UniqueDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A dataset containing the unique elements of an input dataset."""

    def __init__(self, input_dataset, name=None):
        if False:
            i = 10
            return i + 15
        'See `tf.data.Dataset.unique` for details.'
        self._input_dataset = input_dataset
        for ty in nest.flatten(dataset_ops.get_legacy_output_types(input_dataset)):
            if ty not in (dtypes.int32, dtypes.int64, dtypes.string):
                raise TypeError(f'`tf.data.Dataset.unique` does not support type {ty} -- only `tf.int32`, `tf.int64`, and `tf.string` are supported.')
        self._name = name
        variant_tensor = gen_experimental_dataset_ops.unique_dataset(self._input_dataset._variant_tensor, **self._common_args)
        super().__init__(input_dataset, variant_tensor)