"""bincount ops."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('math.bincount', v1=[])
@dispatch.add_dispatch_support
def bincount(arr, weights=None, minlength=None, maxlength=None, dtype=dtypes.int32, name=None, axis=None, binary_output=False):
    if False:
        for i in range(10):
            print('nop')
    "Counts the number of occurrences of each value in an integer array.\n\n  If `minlength` and `maxlength` are not given, returns a vector with length\n  `tf.reduce_max(arr) + 1` if `arr` is non-empty, and length 0 otherwise.\n\n  >>> values = tf.constant([1,1,2,3,2,4,4,5])\n  >>> tf.math.bincount(values)\n  <tf.Tensor: ... numpy=array([0, 2, 2, 1, 2, 1], dtype=int32)>\n\n  Vector length = Maximum element in vector `values` is 5. Adding 1, which is 6\n                  will be the vector length.\n\n  Each bin value in the output indicates number of occurrences of the particular\n  index. Here, index 1 in output has a value 2. This indicates value 1 occurs\n  two times in `values`.\n\n  **Bin-counting with weights**\n\n  >>> values = tf.constant([1,1,2,3,2,4,4,5])\n  >>> weights = tf.constant([1,5,0,1,0,5,4,5])\n  >>> tf.math.bincount(values, weights=weights)\n  <tf.Tensor: ... numpy=array([0, 6, 0, 1, 9, 5], dtype=int32)>\n\n  When `weights` is specified, bins will be incremented by the corresponding\n  weight instead of 1. Here, index 1 in output has a value 6. This is the\n  summation of `weights` corresponding to the value in `values` (i.e. for index\n  1, the first two values are 1 so the first two weights, 1 and 5, are\n  summed).\n\n  There is an equivilance between bin-counting with weights and\n  `unsorted_segement_sum` where `data` is the weights and `segment_ids` are the\n  values.\n\n  >>> values = tf.constant([1,1,2,3,2,4,4,5])\n  >>> weights = tf.constant([1,5,0,1,0,5,4,5])\n  >>> tf.math.unsorted_segment_sum(weights, values, num_segments=6).numpy()\n  array([0, 6, 0, 1, 9, 5], dtype=int32)\n\n  On GPU, `bincount` with weights is only supported when XLA is enabled\n  (typically when a function decorated with `@tf.function(jit_compile=True)`).\n  `unsorted_segment_sum` can be used as a workaround for the non-XLA case on\n  GPU.\n\n  **Bin-counting matrix rows independently**\n\n  This example uses `axis=-1` with a 2 dimensional input and returns a\n  `Tensor` with bincounting where axis 0 is **not** flattened, i.e. an\n  independent bincount for each matrix row.\n\n  >>> data = np.array([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=np.int32)\n  >>> tf.math.bincount(data, axis=-1)\n  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=\n    array([[1, 1, 1, 1],\n           [2, 1, 1, 0]], dtype=int32)>\n\n  **Bin-counting with binary_output**\n\n  This example gives binary output instead of counting the occurrence.\n\n  >>> data = np.array([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=np.int32)\n  >>> tf.math.bincount(data, axis=-1, binary_output=True)\n  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=\n    array([[1, 1, 1, 1],\n           [1, 1, 1, 0]], dtype=int32)>\n\n  **Missing zeros in SparseTensor**\n\n  Note that missing zeros (implict zeros) in SparseTensor are **NOT** counted.\n  This supports cases such as `0` in the values tensor indicates that index/id\n  `0`is present and a missing zero indicates that no index/id is present.\n\n  If counting missing zeros is desired, there are workarounds.\n  For the `axis=0` case, the number of missing zeros can computed by subtracting\n  the number of elements in the SparseTensor's `values` tensor from the\n  number of elements in the dense shape, and this difference can be added to the\n  first element of the output of `bincount`. For all cases, the SparseTensor\n  can be converted to a dense Tensor with `tf.sparse.to_dense` before calling\n  `tf.math.bincount`.\n\n  Args:\n    arr: A Tensor, RaggedTensor, or SparseTensor whose values should be counted.\n      These tensors must have a rank of 2 if `axis=-1`.\n    weights: If non-None, must be the same shape as arr. For each value in\n      `arr`, the bin will be incremented by the corresponding weight instead of\n      1. If non-None, `binary_output` must be False.\n    minlength: If given, ensures the output has length at least `minlength`,\n      padding with zeros at the end if necessary.\n    maxlength: If given, skips values in `arr` that are equal or greater than\n      `maxlength`, ensuring that the output has length at most `maxlength`.\n    dtype: If `weights` is None, determines the type of the output bins.\n    name: A name scope for the associated operations (optional).\n    axis: The axis to slice over. Axes at and below `axis` will be flattened\n      before bin counting. Currently, only `0`, and `-1` are supported. If None,\n      all axes will be flattened (identical to passing `0`).\n    binary_output: If True, this op will output 1 instead of the number of times\n      a token appears (equivalent to one_hot + reduce_any instead of one_hot +\n      reduce_add). Defaults to False.\n\n  Returns:\n    A vector with the same dtype as `weights` or the given `dtype` containing\n    the bincount values.\n\n  Raises:\n    `InvalidArgumentError` if negative values are provided as an input.\n\n  "
    name = 'bincount' if name is None else name
    with ops.name_scope(name):
        arr = tensor_conversion.convert_to_tensor_v2_with_dispatch(arr, name='arr')
        if weights is not None:
            weights = tensor_conversion.convert_to_tensor_v2_with_dispatch(weights, name='weights')
        if weights is not None and binary_output:
            raise ValueError('Arguments `binary_output` and `weights` are mutually exclusive. Please specify only one.')
        if not arr.dtype.is_integer:
            arr = math_ops.cast(arr, dtypes.int32)
        if axis is None:
            axis = 0
        if axis not in [0, -1]:
            raise ValueError(f'Unsupported value for argument axis={axis}. Only 0 and -1 are currently supported.')
        array_is_nonempty = array_ops.size(arr) > 0
        output_size = math_ops.cast(array_is_nonempty, arr.dtype) * (math_ops.reduce_max(arr) + 1)
        if minlength is not None:
            minlength = ops.convert_to_tensor(minlength, name='minlength', dtype=arr.dtype)
            output_size = gen_math_ops.maximum(minlength, output_size)
        if maxlength is not None:
            maxlength = ops.convert_to_tensor(maxlength, name='maxlength', dtype=arr.dtype)
            output_size = gen_math_ops.minimum(maxlength, output_size)
        if axis == 0:
            if weights is not None:
                weights = array_ops.reshape(weights, [-1])
            arr = array_ops.reshape(arr, [-1])
        weights = validate_dense_weights(arr, weights, dtype)
        return gen_math_ops.dense_bincount(input=arr, size=output_size, weights=weights, binary_output=binary_output)

@tf_export(v1=['math.bincount', 'bincount'])
@deprecation.deprecated_endpoints('bincount')
def bincount_v1(arr, weights=None, minlength=None, maxlength=None, dtype=dtypes.int32):
    if False:
        while True:
            i = 10
    'Counts the number of occurrences of each value in an integer array.\n\n  If `minlength` and `maxlength` are not given, returns a vector with length\n  `tf.reduce_max(arr) + 1` if `arr` is non-empty, and length 0 otherwise.\n  If `weights` are non-None, then index `i` of the output stores the sum of the\n  value in `weights` at each index where the corresponding value in `arr` is\n  `i`.\n\n  Args:\n    arr: An int32 tensor of non-negative values.\n    weights: If non-None, must be the same shape as arr. For each value in\n      `arr`, the bin will be incremented by the corresponding weight instead of\n      1.\n    minlength: If given, ensures the output has length at least `minlength`,\n      padding with zeros at the end if necessary.\n    maxlength: If given, skips values in `arr` that are equal or greater than\n      `maxlength`, ensuring that the output has length at most `maxlength`.\n    dtype: If `weights` is None, determines the type of the output bins.\n\n  Returns:\n    A vector with the same dtype as `weights` or the given `dtype`. The bin\n    values.\n  '
    return bincount(arr, weights, minlength, maxlength, dtype)

def validate_dense_weights(values, weights, dtype=None):
    if False:
        i = 10
        return i + 15
    'Validates the passed weight tensor or creates an empty one.'
    if weights is None:
        if dtype:
            return array_ops.constant([], dtype=dtype)
        return array_ops.constant([], dtype=values.dtype)
    if not isinstance(weights, tensor.Tensor):
        raise ValueError(f'Argument `weights` must be a tf.Tensor if `values` is a tf.Tensor. Received weights={weights} of type: {type(weights).__name__}')
    return weights