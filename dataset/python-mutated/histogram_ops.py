"""Histograms.
"""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('histogram_fixed_width_bins')
@dispatch.add_dispatch_support
def histogram_fixed_width_bins(values, value_range, nbins=100, dtype=dtypes.int32, name=None):
    if False:
        while True:
            i = 10
    "Bins the given values for use in a histogram.\n\n  Given the tensor `values`, this operation returns a rank 1 `Tensor`\n  representing the indices of a histogram into which each element\n  of `values` would be binned. The bins are equal width and\n  determined by the arguments `value_range` and `nbins`.\n\n  Args:\n    values:  Numeric `Tensor`.\n    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.\n      values <= value_range[0] will be mapped to hist[0],\n      values >= value_range[1] will be mapped to hist[-1].\n    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.\n    dtype:  dtype for returned histogram.\n    name:  A name for this operation (defaults to 'histogram_fixed_width').\n\n  Returns:\n    A `Tensor` holding the indices of the binned values whose shape matches\n    `values`.\n\n  Raises:\n    TypeError: If any unsupported dtype is provided.\n    tf.errors.InvalidArgumentError: If value_range does not\n        satisfy value_range[0] < value_range[1].\n\n  Examples:\n\n  >>> # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)\n  ...\n  >>> nbins = 5\n  >>> value_range = [0.0, 5.0]\n  >>> new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]\n  >>> indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=5)\n  >>> indices.numpy()\n  array([0, 0, 1, 2, 4, 4], dtype=int32)\n  "
    with ops.name_scope(name, 'histogram_fixed_width_bins', [values, value_range, nbins]):
        values = ops.convert_to_tensor(values, name='values')
        shape = array_ops.shape(values)
        values = array_ops.reshape(values, [-1])
        value_range = ops.convert_to_tensor(value_range, name='value_range')
        nbins = ops.convert_to_tensor(nbins, dtype=dtypes.int32, name='nbins')
        check = control_flow_assert.Assert(math_ops.greater(nbins, 0), ['nbins %s must > 0' % nbins])
        nbins = control_flow_ops.with_dependencies([check], nbins)
        nbins_float = math_ops.cast(nbins, values.dtype)
        scaled_values = math_ops.truediv(values - value_range[0], value_range[1] - value_range[0], name='scaled_values')
        indices = math_ops.floor(nbins_float * scaled_values, name='indices')
        indices = math_ops.cast(clip_ops.clip_by_value(indices, 0, nbins_float - 1), dtypes.int32)
        return array_ops.reshape(indices, shape)

@tf_export('histogram_fixed_width')
@dispatch.add_dispatch_support
def histogram_fixed_width(values, value_range, nbins=100, dtype=dtypes.int32, name=None):
    if False:
        print('Hello World!')
    "Return histogram of values.\n\n  Given the tensor `values`, this operation returns a rank 1 histogram counting\n  the number of entries in `values` that fell into every bin.  The bins are\n  equal width and determined by the arguments `value_range` and `nbins`.\n\n  Args:\n    values:  Numeric `Tensor`.\n    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.\n      values <= value_range[0] will be mapped to hist[0],\n      values >= value_range[1] will be mapped to hist[-1].\n    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.\n    dtype:  dtype for returned histogram.\n    name:  A name for this operation (defaults to 'histogram_fixed_width').\n\n  Returns:\n    A 1-D `Tensor` holding histogram of values.\n\n  Raises:\n    TypeError: If any unsupported dtype is provided.\n    tf.errors.InvalidArgumentError: If value_range does not\n        satisfy value_range[0] < value_range[1].\n\n  Examples:\n\n  >>> # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)\n  ...\n  >>> nbins = 5\n  >>> value_range = [0.0, 5.0]\n  >>> new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]\n  >>> hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)\n  >>> hist.numpy()\n  array([2, 1, 1, 0, 2], dtype=int32)\n  "
    with ops.name_scope(name, 'histogram_fixed_width', [values, value_range, nbins]) as name:
        return gen_math_ops._histogram_fixed_width(values, value_range, nbins, dtype=dtype, name=name)