"""The implementation of `tf.data.Dataset.unbatch`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops

def _unbatch(input_dataset, name=None):
    if False:
        return 10
    'See `Dataset.unbatch()` for details.'
    normalized_dataset = dataset_ops.normalize_to_dense(input_dataset)
    return _UnbatchDataset(normalized_dataset, name=name)

class _UnbatchDataset(dataset_ops.UnaryDataset):
    """A dataset that splits the elements of its input into multiple elements."""

    def __init__(self, input_dataset, name=None):
        if False:
            for i in range(10):
                print('nop')
        'See `unbatch()` for more details.'
        flat_shapes = input_dataset._flat_shapes
        if any((s.ndims == 0 for s in flat_shapes)):
            raise ValueError('Cannot unbatch an input with scalar components.')
        known_batch_dim = tensor_shape.Dimension(None)
        for s in flat_shapes:
            try:
                known_batch_dim = known_batch_dim.merge_with(s[0])
            except ValueError as e:
                raise ValueError(f'`unbatch()` is only supported for datasets of elements whose components have a matching leading dimension. Encountered both {known_batch_dim} and {s[0]}.') from e
        self._input_dataset = input_dataset
        self._structure = nest.map_structure(lambda component_spec: component_spec._unbatch(), dataset_ops.get_structure(input_dataset))
        self._name = name
        variant_tensor = ged_ops.unbatch_dataset(self._input_dataset._variant_tensor, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        if False:
            print('Hello World!')
        return self._structure