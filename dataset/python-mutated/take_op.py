"""The implementation of `tf.data.Dataset.skip`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops

def _take(self, count, name=None):
    if False:
        i = 10
        return i + 15
    return _TakeDataset(self, count, name=name)

class _TakeDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` containing the first `count` elements from its input."""

    def __init__(self, input_dataset, count, name=None):
        if False:
            i = 10
            return i + 15
        'See `Dataset.take()` for details.'
        self._input_dataset = input_dataset
        self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name='count')
        self._name = name
        variant_tensor = gen_dataset_ops.take_dataset(input_dataset._variant_tensor, count=self._count, **self._common_args)
        super().__init__(input_dataset, variant_tensor)