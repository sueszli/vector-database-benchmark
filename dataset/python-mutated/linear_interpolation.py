"""Linear interpolation method."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
__all__ = ['interpolate']

def interpolate(x: types.RealTensor, x_data: types.RealTensor, y_data: types.RealTensor, left_slope: types.RealTensor=None, right_slope: types.RealTensor=None, validate_args: bool=False, optimize_for_tpu: bool=False, dtype: tf.DType=None, name: str=None):
    if False:
        print('Hello World!')
    "Performs linear interpolation for supplied points.\n\n  Given a set of knots whose x- and y- coordinates are in `x_data` and `y_data`,\n  this function returns y-values for x-coordinates in `x` via piecewise\n  linear interpolation.\n\n  `x_data` must be non decreasing, but `y_data` don't need to be because we do\n  not require the function approximated by these knots to be monotonic.\n\n  #### Examples\n\n  ```python\n  import tf_quant_finance as tff\n  x = [-10, -1, 1, 3, 6, 7, 8, 15, 18, 25, 30, 35]\n  x_data = [-1, 2, 6, 8, 18, 30.0]\n  y_data = [10, -1, -5, 7, 9, 20]\n\n  tff.math.interpolation.linear.interpolate(x, x_data, y_data,\n                                            dtype=tf.float64)\n  # Expected: [ 10, 10, 2.66666667, -2, -5, 1, 7, 8.4, 9, 15.41666667, 20, 20]\n  ```\n\n  Args:\n    x: x-coordinates for which we need to get interpolation. A N-D\n      `Tensor` of real dtype. First N-1 dimensions represent batching\n      dimensions.\n    x_data: x coordinates. A N-D `Tensor` of real dtype. Should be sorted\n      in non decreasing order. First N-1 dimensions represent batching\n      dimensions.\n    y_data: y coordinates. A N-D `Tensor` of real dtype. Should have the\n      compatible shape as `x_data`. First N-1 dimensions represent batching\n      dimensions.\n    left_slope: The slope to use for extrapolation with x-coordinate smaller\n      than the min `x_data`. It's a 0-D or N-D `Tensor`.\n      Default value: `None`, which maps to `0.0` meaning constant extrapolation,\n      i.e. extrapolated value will be the leftmost `y_data`.\n    right_slope: The slope to use for extrapolation with x-coordinate greater\n      than the max `x_data`. It's a 0-D or N-D `Tensor`.\n      Default value: `None` which maps to `0.0` meaning constant extrapolation,\n      i.e. extrapolated value will be the rightmost `y_data`.\n    validate_args: Python `bool` that indicates whether the function performs\n      the check if the shapes of `x_data` and `y_data` are equal and that the\n      elements in `x_data` are non decreasing. If this value is set to `False`\n      and the elements in `x_data` are not increasing, the result of linear\n      interpolation may be wrong.\n      Default value: `False`.\n    optimize_for_tpu: A Python bool. If `True`, the algorithm uses one-hot\n      encoding to lookup indices of `x` in `x_data`. This significantly\n      improves performance of the algorithm on a TPU device but may slow down\n      performance on the CPU.\n      Default value: `False`.\n    dtype: Optional tf.dtype for `x`, x_data`, `y_data`, `left_slope` and\n      `right_slope`.\n      Default value: `None` which means that the `dtype` inferred from\n        `x`.\n    name: Python str. The name prefixed to the ops created by this function.\n      Default value: `None` which maps to 'linear_interpolation'.\n\n  Returns:\n    A N-D `Tensor` of real dtype corresponding to the x-values in `x`.\n  "
    name = name or 'linear_interpolate'
    with tf.name_scope(name):
        x = tf.convert_to_tensor(x, dtype=dtype, name='x')
        dtype = dtype or x.dtype
        x_data = tf.convert_to_tensor(x_data, dtype=dtype, name='x_data')
        y_data = tf.convert_to_tensor(y_data, dtype=dtype, name='y_data')
        (x, x_data, y_data) = tff_utils.broadcast_common_batch_shape(x, x_data, y_data)
        batch_rank = x.shape.rank - 1
        if batch_rank == 0:
            x = tf.expand_dims(x, 0)
            x_data = tf.expand_dims(x_data, 0)
            y_data = tf.expand_dims(y_data, 0)
        if left_slope is None:
            left_slope = tf.constant(0.0, dtype=x.dtype, name='left_slope')
        else:
            left_slope = tf.convert_to_tensor(left_slope, dtype=dtype, name='left_slope')
        if right_slope is None:
            right_slope = tf.constant(0.0, dtype=x.dtype, name='right_slope')
        else:
            right_slope = tf.convert_to_tensor(right_slope, dtype=dtype, name='right_slope')
        control_deps = []
        if validate_args:
            diffs = x_data[..., 1:] - x_data[..., :-1]
            assertion = tf.debugging.assert_greater_equal(diffs, tf.zeros_like(diffs), message='x_data is not sorted in non-decreasing order.')
            control_deps.append(assertion)
            control_deps.append(tf.compat.v1.assert_equal(tff_utils.get_shape(x_data), tff_utils.get_shape(y_data)))
        with tf.control_dependencies(control_deps):
            upper_indices = tf.searchsorted(x_data, x, side='left', out_type=tf.int32)
            x_data_size = tff_utils.get_shape(x_data)[-1]
            at_min = tf.equal(upper_indices, 0)
            at_max = tf.equal(upper_indices, x_data_size)
            values_min = tf.expand_dims(y_data[..., 0], -1) + left_slope * (x - tf.broadcast_to(tf.expand_dims(x_data[..., 0], -1), shape=tff_utils.get_shape(x)))
            values_max = tf.expand_dims(y_data[..., -1], -1) + right_slope * (x - tf.broadcast_to(tf.expand_dims(x_data[..., -1], -1), shape=tff_utils.get_shape(x)))
            lower_encoding = tf.math.maximum(upper_indices - 1, 0)
            upper_encoding = tf.math.minimum(upper_indices, x_data_size - 1)
            if optimize_for_tpu:
                lower_encoding = tf.one_hot(lower_encoding, x_data_size, dtype=dtype)
                upper_encoding = tf.one_hot(upper_encoding, x_data_size, dtype=dtype)

            def get_slice(x, encoding):
                if False:
                    for i in range(10):
                        print('nop')
                if optimize_for_tpu:
                    return tf.math.reduce_sum(tf.expand_dims(x, axis=-2) * encoding, axis=-1)
                else:
                    return tf.gather(x, encoding, axis=-1, batch_dims=x.shape.rank - 1)
            x_data_lower = get_slice(x_data, lower_encoding)
            x_data_upper = get_slice(x_data, upper_encoding)
            y_data_lower = get_slice(y_data, lower_encoding)
            y_data_upper = get_slice(y_data, upper_encoding)
            x_data_diff = x_data_upper - x_data_lower
            floor_x_diff = tf.where(at_min | at_max, x_data_diff + 1, x_data_diff)
            interpolated = y_data_lower + (x - x_data_lower) * tf.math.divide_no_nan(y_data_upper - y_data_lower, floor_x_diff)
            interpolated = tf.where(at_min, values_min, interpolated)
            interpolated = tf.where(at_max, values_max, interpolated)
            if batch_rank > 0:
                return interpolated
            else:
                return tf.squeeze(interpolated, 0)