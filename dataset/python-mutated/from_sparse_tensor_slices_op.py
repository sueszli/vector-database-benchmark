"""The implementation of `tf.data.Dataset.from_sparse_tensor_slices`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops

def _from_sparse_tensor_slices(sparse_tensor):
    if False:
        while True:
            i = 10
    return dataset_ops.DatasetV1Adapter(_SparseTensorSliceDataset(sparse_tensor))

class _SparseTensorSliceDataset(dataset_ops.DatasetSource):
    """A `Dataset` that splits a rank-N `tf.sparse.SparseTensor` into its rows."""

    def __init__(self, sparse_tensor):
        if False:
            print('Hello World!')
        'See `Dataset.from_sparse_tensor_slices()` for details.'
        if not isinstance(sparse_tensor, sparse_tensor_lib.SparseTensor):
            raise TypeError(f'Invalid `sparse_tensor`. `sparse_tensor` must be a `tf.sparse.SparseTensor`. Got {type(sparse_tensor)}.')
        self._sparse_tensor = sparse_tensor
        indices_shape = self._sparse_tensor.indices.get_shape()
        shape_shape = self._sparse_tensor.dense_shape.get_shape()
        rank = (indices_shape.dims[1] - 1).merge_with(shape_shape.dims[0] - 1)
        self._structure = (tensor_spec.TensorSpec([None, rank], dtypes.int64), tensor_spec.TensorSpec([None], self._sparse_tensor.dtype), tensor_spec.TensorSpec([rank], dtypes.int64))
        variant_tensor = gen_dataset_ops.sparse_tensor_slice_dataset(self._sparse_tensor.indices, self._sparse_tensor.values, self._sparse_tensor.dense_shape)
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        return self._structure