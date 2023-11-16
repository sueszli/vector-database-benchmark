"""The implementation of `tf.data.Dataset.range`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops

def _range(*args, **kwargs):
    if False:
        while True:
            i = 10
    return _RangeDataset(*args, **kwargs)

class _RangeDataset(dataset_ops.DatasetSource):
    """A `Dataset` of a step separated range of values."""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'See `Dataset.range()` for details.'
        self._parse_args(*args, **kwargs)
        self._structure = tensor_spec.TensorSpec([], self._output_type)
        variant_tensor = gen_dataset_ops.range_dataset(start=self._start, stop=self._stop, step=self._step, **self._common_args)
        super().__init__(variant_tensor)

    def _parse_args(self, *args, **kwargs):
        if False:
            return 10
        'Parses arguments according to the same rules as the `range()` builtin.'
        if len(args) == 1:
            self._start = self._build_tensor(0, 'start')
            self._stop = self._build_tensor(args[0], 'stop')
            self._step = self._build_tensor(1, 'step')
        elif len(args) == 2:
            self._start = self._build_tensor(args[0], 'start')
            self._stop = self._build_tensor(args[1], 'stop')
            self._step = self._build_tensor(1, 'step')
        elif len(args) == 3:
            self._start = self._build_tensor(args[0], 'start')
            self._stop = self._build_tensor(args[1], 'stop')
            self._step = self._build_tensor(args[2], 'step')
        else:
            raise ValueError(f'Invalid `args`. The length of `args` should be between 1 and 3 but was {len(args)}.')
        if 'output_type' in kwargs:
            self._output_type = kwargs['output_type']
        else:
            self._output_type = dtypes.int64
        self._name = kwargs['name'] if 'name' in kwargs else None

    def _build_tensor(self, int64_value, name):
        if False:
            while True:
                i = 10
        return ops.convert_to_tensor(int64_value, dtype=dtypes.int64, name=name)

    @property
    def element_spec(self):
        if False:
            for i in range(10):
                print('nop')
        return self._structure