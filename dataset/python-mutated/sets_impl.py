"""Implementation of tf.sets."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_set_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
_VALID_DTYPES = frozenset([dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.string])

@tf_export('sets.size', v1=['sets.size', 'sets.set_size'])
@dispatch.add_dispatch_support
def set_size(a, validate_indices=True):
    if False:
        print('Hello World!')
    'Compute number of unique elements along last dimension of `a`.\n\n  Args:\n    a: `SparseTensor`, with indices sorted in row-major order.\n    validate_indices: Whether to validate the order and range of sparse indices\n      in `a`. Note that setting this to `false` allows for undefined behavior\n      when calling this function with invalid indices.\n\n  Returns:\n    `int32` `Tensor` of set sizes. For `a` ranked `n`, this is a `Tensor` with\n    rank `n-1`, and the same 1st `n-1` dimensions as `a`. Each value is the\n    number of unique elements in the corresponding `[0...n-1]` dimension of `a`.\n\n  Raises:\n    TypeError: If `a` is an invalid types.\n  '
    a = sparse_tensor.convert_to_tensor_or_sparse_tensor(a, name='a')
    if not isinstance(a, sparse_tensor.SparseTensor):
        raise TypeError('Expected `SparseTensor`, got %s.' % a)
    if a.values.dtype.base_dtype not in _VALID_DTYPES:
        raise TypeError(f'Invalid dtype `{a.values.dtype}` not in supported dtypes: `{_VALID_DTYPES}`.')
    return gen_set_ops.set_size(a.indices, a.values, a.dense_shape, validate_indices)
ops.NotDifferentiable('SetSize')
ops.NotDifferentiable('DenseToDenseSetOperation')
ops.NotDifferentiable('DenseToSparseSetOperation')
ops.NotDifferentiable('SparseToSparseSetOperation')

def _convert_to_tensors_or_sparse_tensors(a, b):
    if False:
        i = 10
        return i + 15
    'Convert to tensor types, and flip order if necessary.\n\n  Args:\n    a: `Tensor` or `SparseTensor` of the same type as `b`.\n    b: `Tensor` or `SparseTensor` of the same type as `a`.\n\n  Returns:\n    Tuple of `(a, b, flipped)`, where `a` and `b` have been converted to\n    `Tensor` or `SparseTensor`, and `flipped` indicates whether the order has\n    been flipped to make it dense,sparse instead of sparse,dense (since the set\n    ops do not support the latter).\n  '
    a = sparse_tensor.convert_to_tensor_or_sparse_tensor(a, name='a')
    if a.dtype.base_dtype not in _VALID_DTYPES:
        raise TypeError(f"'a' has invalid dtype `{a.dtype}` not in supported dtypes: `{_VALID_DTYPES}`.")
    b = sparse_tensor.convert_to_tensor_or_sparse_tensor(b, name='b')
    if b.dtype.base_dtype != a.dtype.base_dtype:
        raise TypeError("Types don't match, %s vs %s." % (a.dtype, b.dtype))
    if isinstance(a, sparse_tensor.SparseTensor) and (not isinstance(b, sparse_tensor.SparseTensor)):
        return (b, a, True)
    return (a, b, False)

def _set_operation(a, b, set_operation, validate_indices=True):
    if False:
        while True:
            i = 10
    'Compute set operation of elements in last dimension of `a` and `b`.\n\n  All but the last dimension of `a` and `b` must match.\n\n  Args:\n    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices\n      must be sorted in row-major order.\n    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be\n      `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be sorted\n      in row-major order.\n    set_operation: String indicating set operation. See\n        SetOperationOp::SetOperationFromContext for valid values.\n    validate_indices: Whether to validate the order and range of sparse indices\n      in `a` and `b`.\n\n  Returns:\n    A `SparseTensor` with the same rank as `a` and `b`, and all but the last\n    dimension the same. Elements along the last dimension contain the results\n    of the set operation.\n\n  Raises:\n    TypeError: If inputs are invalid types.\n    ValueError: If `a` is sparse and `b` is dense.\n  '
    if isinstance(a, sparse_tensor.SparseTensor):
        if isinstance(b, sparse_tensor.SparseTensor):
            (indices, values, shape) = gen_set_ops.sparse_to_sparse_set_operation(a.indices, a.values, a.dense_shape, b.indices, b.values, b.dense_shape, set_operation, validate_indices)
        else:
            raise ValueError('Sparse,Dense is not supported, but Dense,Sparse is. Please flip the order of your inputs.')
    elif isinstance(b, sparse_tensor.SparseTensor):
        (indices, values, shape) = gen_set_ops.dense_to_sparse_set_operation(a, b.indices, b.values, b.dense_shape, set_operation, validate_indices)
    else:
        (indices, values, shape) = gen_set_ops.dense_to_dense_set_operation(a, b, set_operation, validate_indices)
    return sparse_tensor.SparseTensor(indices, values, shape)

@tf_export('sets.intersection', v1=['sets.intersection', 'sets.set_intersection'])
@dispatch.add_dispatch_support
def set_intersection(a, b, validate_indices=True):
    if False:
        i = 10
        return i + 15
    'Compute set intersection of elements in last dimension of `a` and `b`.\n\n  All but the last dimension of `a` and `b` must match.\n\n  Example:\n\n  ```python\n    import tensorflow as tf\n    import collections\n\n    # Represent the following array of sets as a sparse tensor:\n    # a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])\n    a = collections.OrderedDict([\n        ((0, 0, 0), 1),\n        ((0, 0, 1), 2),\n        ((0, 1, 0), 3),\n        ((1, 0, 0), 4),\n        ((1, 1, 0), 5),\n        ((1, 1, 1), 6),\n    ])\n    a = tf.sparse.SparseTensor(list(a.keys()), list(a.values()),\n                               dense_shape=[2,2,2])\n\n    # b = np.array([[{1}, {}], [{4}, {5, 6, 7, 8}]])\n    b = collections.OrderedDict([\n        ((0, 0, 0), 1),\n        ((1, 0, 0), 4),\n        ((1, 1, 0), 5),\n        ((1, 1, 1), 6),\n        ((1, 1, 2), 7),\n        ((1, 1, 3), 8),\n    ])\n    b = tf.sparse.SparseTensor(list(b.keys()), list(b.values()),\n                               dense_shape=[2, 2, 4])\n\n    # `tf.sets.intersection` is applied to each aligned pair of sets.\n    tf.sets.intersection(a, b)\n\n    # The result will be equivalent to either of:\n    #\n    # np.array([[{1}, {}], [{4}, {5, 6}]])\n    #\n    # collections.OrderedDict([\n    #     ((0, 0, 0), 1),\n    #     ((1, 0, 0), 4),\n    #     ((1, 1, 0), 5),\n    #     ((1, 1, 1), 6),\n    # ])\n  ```\n\n  Args:\n    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices\n      must be sorted in row-major order.\n    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices\n      must be sorted in row-major order.\n    validate_indices: Whether to validate the order and range of sparse indices\n      in `a` and `b`.\n\n  Returns:\n    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but\n    the last dimension the same. Elements along the last dimension contain the\n    intersections.\n  '
    (a, b, _) = _convert_to_tensors_or_sparse_tensors(a, b)
    return _set_operation(a, b, 'intersection', validate_indices)

@tf_export('sets.difference', v1=['sets.difference', 'sets.set_difference'])
@dispatch.add_dispatch_support
def set_difference(a, b, aminusb=True, validate_indices=True):
    if False:
        for i in range(10):
            print('nop')
    'Compute set difference of elements in last dimension of `a` and `b`.\n\n  All but the last dimension of `a` and `b` must match.\n\n  Example:\n\n  ```python\n    import tensorflow as tf\n    import collections\n\n    # Represent the following array of sets as a sparse tensor:\n    # a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])\n    a = collections.OrderedDict([\n        ((0, 0, 0), 1),\n        ((0, 0, 1), 2),\n        ((0, 1, 0), 3),\n        ((1, 0, 0), 4),\n        ((1, 1, 0), 5),\n        ((1, 1, 1), 6),\n    ])\n    a = tf.sparse.SparseTensor(list(a.keys()), list(a.values()),\n                               dense_shape=[2, 2, 2])\n\n    # np.array([[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]])\n    b = collections.OrderedDict([\n        ((0, 0, 0), 1),\n        ((0, 0, 1), 3),\n        ((0, 1, 0), 2),\n        ((1, 0, 0), 4),\n        ((1, 0, 1), 5),\n        ((1, 1, 0), 5),\n        ((1, 1, 1), 6),\n        ((1, 1, 2), 7),\n        ((1, 1, 3), 8),\n    ])\n    b = tf.sparse.SparseTensor(list(b.keys()), list(b.values()),\n                               dense_shape=[2, 2, 4])\n\n    # `set_difference` is applied to each aligned pair of sets.\n    tf.sets.difference(a, b)\n\n    # The result will be equivalent to either of:\n    #\n    # np.array([[{2}, {3}], [{}, {}]])\n    #\n    # collections.OrderedDict([\n    #     ((0, 0, 0), 2),\n    #     ((0, 1, 0), 3),\n    # ])\n  ```\n\n  Args:\n    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices\n      must be sorted in row-major order.\n    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices\n      must be sorted in row-major order.\n    aminusb: Whether to subtract `b` from `a`, vs vice versa.\n    validate_indices: Whether to validate the order and range of sparse indices\n      in `a` and `b`.\n\n  Returns:\n    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but\n    the last dimension the same. Elements along the last dimension contain the\n    differences.\n\n  Raises:\n    TypeError: If inputs are invalid types, or if `a` and `b` have\n        different types.\n    ValueError: If `a` is sparse and `b` is dense.\n    errors_impl.InvalidArgumentError: If the shapes of `a` and `b` do not\n        match in any dimension other than the last dimension.\n  '
    (a, b, flipped) = _convert_to_tensors_or_sparse_tensors(a, b)
    if flipped:
        aminusb = not aminusb
    return _set_operation(a, b, 'a-b' if aminusb else 'b-a', validate_indices)

@tf_export('sets.union', v1=['sets.union', 'sets.set_union'])
@dispatch.add_dispatch_support
def set_union(a, b, validate_indices=True):
    if False:
        i = 10
        return i + 15
    'Compute set union of elements in last dimension of `a` and `b`.\n\n  All but the last dimension of `a` and `b` must match.\n\n  Example:\n\n  ```python\n    import tensorflow as tf\n    import collections\n\n    # [[{1, 2}, {3}], [{4}, {5, 6}]]\n    a = collections.OrderedDict([\n        ((0, 0, 0), 1),\n        ((0, 0, 1), 2),\n        ((0, 1, 0), 3),\n        ((1, 0, 0), 4),\n        ((1, 1, 0), 5),\n        ((1, 1, 1), 6),\n    ])\n    a = tf.sparse.SparseTensor(list(a.keys()), list(a.values()),\n                               dense_shape=[2, 2, 2])\n\n    # [[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]]\n    b = collections.OrderedDict([\n        ((0, 0, 0), 1),\n        ((0, 0, 1), 3),\n        ((0, 1, 0), 2),\n        ((1, 0, 0), 4),\n        ((1, 0, 1), 5),\n        ((1, 1, 0), 5),\n        ((1, 1, 1), 6),\n        ((1, 1, 2), 7),\n        ((1, 1, 3), 8),\n    ])\n    b = tf.sparse.SparseTensor(list(b.keys()), list(b.values()),\n                               dense_shape=[2, 2, 4])\n\n    # `set_union` is applied to each aligned pair of sets.\n    tf.sets.union(a, b)\n\n    # The result will be a equivalent to either of:\n    #\n    # np.array([[{1, 2, 3}, {2, 3}], [{4, 5}, {5, 6, 7, 8}]])\n    #\n    # collections.OrderedDict([\n    #     ((0, 0, 0), 1),\n    #     ((0, 0, 1), 2),\n    #     ((0, 0, 2), 3),\n    #     ((0, 1, 0), 2),\n    #     ((0, 1, 1), 3),\n    #     ((1, 0, 0), 4),\n    #     ((1, 0, 1), 5),\n    #     ((1, 1, 0), 5),\n    #     ((1, 1, 1), 6),\n    #     ((1, 1, 2), 7),\n    #     ((1, 1, 3), 8),\n    # ])\n  ```\n\n  Args:\n    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices\n      must be sorted in row-major order.\n    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices\n      must be sorted in row-major order.\n    validate_indices: Whether to validate the order and range of sparse indices\n      in `a` and `b`.\n\n  Returns:\n    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but\n    the last dimension the same. Elements along the last dimension contain the\n    unions.\n  '
    (a, b, _) = _convert_to_tensors_or_sparse_tensors(a, b)
    return _set_operation(a, b, 'union', validate_indices)