"""The implementation of `tf.data.Dataset.skip`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops

def _skip(self, count, name=None):
    if False:
        for i in range(10):
            print('nop')
    return _SkipDataset(self, count, name)

class _SkipDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` skipping the first `count` elements from its input."""

    def __init__(self, input_dataset, count, name=None):
        if False:
            print('Hello World!')
        'See `Dataset.skip()` for details.'
        self._input_dataset = input_dataset
        self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name='count')
        self._name = name
        variant_tensor = gen_dataset_ops.skip_dataset(input_dataset._variant_tensor, count=self._count, **self._common_args)
        super().__init__(input_dataset, variant_tensor)