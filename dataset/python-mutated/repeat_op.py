"""The implementation of `tf.data.Dataset.repeat`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops

def _repeat(input_dataset, count, name):
    if False:
        return 10
    return _RepeatDataset(input_dataset, count, name)

class _RepeatDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that repeats its input several times."""

    def __init__(self, input_dataset, count, name=None):
        if False:
            return 10
        'See `Dataset.repeat()` for details.'
        self._input_dataset = input_dataset
        if count is None:
            self._count = constant_op.constant(-1, dtype=dtypes.int64, name='count')
        else:
            self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name='count')
        self._name = name
        variant_tensor = gen_dataset_ops.repeat_dataset(input_dataset._variant_tensor, count=self._count, **self._common_args)
        super().__init__(input_dataset, variant_tensor)