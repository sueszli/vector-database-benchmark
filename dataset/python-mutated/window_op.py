"""The implementation of `tf.data.Dataset.window`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops

def _window(input_dataset, size, shift, stride, drop_remainder, name):
    if False:
        print('Hello World!')
    if shift is None:
        shift = size
    return _WindowDataset(input_dataset, size, shift, stride, drop_remainder, name=name)

class _WindowDataset(dataset_ops.UnaryDataset):
    """A dataset that creates window datasets from the input elements."""

    def __init__(self, input_dataset, size, shift, stride, drop_remainder, name=None):
        if False:
            print('Hello World!')
        'See `window()` for more details.'
        self._input_dataset = input_dataset
        self._size = ops.convert_to_tensor(size, dtype=dtypes.int64, name='size')
        self._shift = ops.convert_to_tensor(shift, dtype=dtypes.int64, name='shift')
        self._stride = ops.convert_to_tensor(stride, dtype=dtypes.int64, name='stride')
        self._drop_remainder = ops.convert_to_tensor(drop_remainder, dtype=dtypes.bool, name='drop_remainder')
        self._structure = nest.pack_sequence_as(dataset_ops.get_legacy_output_classes(input_dataset), [dataset_ops.DatasetSpec(structure.convert_legacy_structure(output_type, output_shape, output_class)) for (output_class, output_shape, output_type) in zip(nest.flatten(dataset_ops.get_legacy_output_classes(input_dataset)), nest.flatten(dataset_ops.get_legacy_output_shapes(input_dataset)), nest.flatten(dataset_ops.get_legacy_output_types(input_dataset)))])
        self._name = name
        variant_tensor = gen_dataset_ops.window_dataset(input_dataset._variant_tensor, size=self._size, shift=self._shift, stride=self._stride, drop_remainder=self._drop_remainder, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        return self._structure