"""The implementation of `tf.data.Dataset.batch`."""
import warnings
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops

def _batch(input_dataset, batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None, name=None):
    if False:
        return 10
    'See `Dataset.batch` for details.'
    if num_parallel_calls is None or debug_mode.DEBUG_MODE:
        if deterministic is not None and (not debug_mode.DEBUG_MODE):
            warnings.warn('The `deterministic` argument has no effect unless the `num_parallel_calls` argument is specified.')
        return _BatchDataset(input_dataset, batch_size, drop_remainder, name=name)
    else:
        return _ParallelBatchDataset(input_dataset, batch_size, drop_remainder, num_parallel_calls, deterministic, name=name)

class _BatchDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that batches contiguous elements from its input."""

    def __init__(self, input_dataset, batch_size, drop_remainder, name=None):
        if False:
            i = 10
            return i + 15
        'See `Dataset.batch()` for details.'
        self._input_dataset = input_dataset
        self._batch_size = ops.convert_to_tensor(batch_size, dtype=dtypes.int64, name='batch_size')
        self._drop_remainder = ops.convert_to_tensor(drop_remainder, dtype=dtypes.bool, name='drop_remainder')
        constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
        if constant_drop_remainder:
            constant_batch_size = tensor_util.constant_value(self._batch_size)
            self._structure = nest.map_structure(lambda component_spec: component_spec._batch(constant_batch_size), input_dataset.element_spec)
        else:
            self._structure = nest.map_structure(lambda component_spec: component_spec._batch(None), input_dataset.element_spec)
        self._name = name
        variant_tensor = gen_dataset_ops.batch_dataset_v2(input_dataset._variant_tensor, batch_size=self._batch_size, drop_remainder=self._drop_remainder, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        if False:
            while True:
                i = 10
        return self._structure

class _ParallelBatchDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that batches contiguous elements from its input in parallel."""

    def __init__(self, input_dataset, batch_size, drop_remainder, num_parallel_calls, deterministic, name=None):
        if False:
            return 10
        'See `Dataset.batch()` for details.'
        self._input_dataset = input_dataset
        self._batch_size = ops.convert_to_tensor(batch_size, dtype=dtypes.int64, name='batch_size')
        self._drop_remainder = ops.convert_to_tensor(drop_remainder, dtype=dtypes.bool, name='drop_remainder')
        self._num_parallel_calls = ops.convert_to_tensor(num_parallel_calls, dtype=dtypes.int64, name='num_parallel_calls')
        if deterministic is None:
            self._deterministic = 'default'
        elif deterministic:
            self._deterministic = 'true'
        else:
            self._deterministic = 'false'
        constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
        if constant_drop_remainder:
            constant_batch_size = tensor_util.constant_value(self._batch_size)
            self._structure = nest.map_structure(lambda component_spec: component_spec._batch(constant_batch_size), input_dataset.element_spec)
        else:
            self._structure = nest.map_structure(lambda component_spec: component_spec._batch(None), input_dataset.element_spec)
        self._name = name
        variant_tensor = gen_dataset_ops.parallel_batch_dataset(input_dataset._variant_tensor, batch_size=self._batch_size, num_parallel_calls=self._num_parallel_calls, drop_remainder=self._drop_remainder, deterministic=self._deterministic, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        if False:
            print('Hello World!')
        return self._structure