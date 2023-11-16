"""The implementation of `tf.data.Dataset.ragged_batch`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops.ragged import ragged_tensor

def _ragged_batch(input_dataset, batch_size, drop_remainder=False, row_splits_dtype=dtypes.int64, name=None):
    if False:
        return 10
    ragged_dataset = _DenseToRaggedDataset(input_dataset, row_splits_dtype, name)
    return ragged_dataset.batch(batch_size, drop_remainder)

class _DenseToRaggedDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that encodes dense inputs as ragged (w/ ragged_rank=0).

  In particular:

  * Any tf.Tensor elements with rank>0 are encoded as ragged tensors with
    ragged_rank=0.  This allows tensors with varying shape to be batched
    together.
  * Any other elements are left as-is.
  """

    def __init__(self, input_dataset, row_splits_dtype, name=None):
        if False:
            return 10
        'Constructs a new _DenseToRaggedDataset.\n\n    Args:\n      input_dataset: The dataset whose tf.Tensor elements should be made ragged.\n      row_splits_dtype: The dtype that should be used for the `row_splits` of\n        any new ragged tensors.  Existing `tf.RaggedTensor` elements do *not*\n        have their row_splits dtype changed.\n      name: (Optional.) A string indicating a name for the `tf.data` operation.\n    '

        def to_ragged_spec(spec):
            if False:
                for i in range(10):
                    print('nop')
            'Returns the new spec based on RaggedTensors.'
            if not isinstance(spec, tensor.TensorSpec) or spec.shape.rank is None or spec.shape.is_fully_defined():
                return spec
            else:
                ragged_rank = max([axis for (axis, size) in enumerate(spec.shape.as_list()) if size is None])
                return ragged_tensor.RaggedTensorSpec(shape=spec.shape, dtype=spec.dtype, ragged_rank=ragged_rank, row_splits_dtype=row_splits_dtype)
        self._structure = nest.map_structure(to_ragged_spec, input_dataset.element_spec)

        def to_ragged_variant(value):
            if False:
                return 10
            'Re-encode Tensors as RaggedTensors.'
            if not isinstance(value, tensor.Tensor) or value.shape.rank is None or value.shape.is_fully_defined():
                return value
            else:
                spec = to_ragged_spec(tensor.TensorSpec.from_tensor(value))
                if spec._ragged_rank > 0:
                    value = ragged_tensor.RaggedTensor.from_tensor(value, ragged_rank=spec._ragged_rank)
                return spec._to_tensor_list(value)[0]
        if structured_function._should_unpack(input_dataset.element_spec):
            map_fn = lambda *value: nest.map_structure(to_ragged_variant, value)
        else:
            map_fn = lambda value: nest.map_structure(to_ragged_variant, value)
        self._mapped_dataset = input_dataset.map(map_fn)
        self._name = name
        variant = self._mapped_dataset._variant_tensor
        super().__init__(input_dataset, variant)

    @property
    def element_spec(self):
        if False:
            print('Hello World!')
        return self._structure