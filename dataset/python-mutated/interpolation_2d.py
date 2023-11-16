"""Interpolation functions in a 2-dimensional space."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.math.interpolation import cubic
from tf_quant_finance.math.interpolation import linear

class Interpolation2D:
    """Performs interpolation in a 2-dimensional space.

  For input `x_data` in x-direction we assume that values in y-direction are
  given by `y_data` and the corresponding function values by `z_data`.
  For given `x` and `y` along x- and y- direction respectively,
  the interpolated function values are computed on grid `[x, y]`.
  The interpolation is first performed along y-direction for every `x_data`
  point and all `y` using 1-d cubic spline interpolation. Next, for
  each interpolated `y_value` point, the function values are interpolated along
  x-direction for `x` using 1-d cubic spline interpolation.
  Constant extrapolation is used for the linear interpolation and natural
  boundary conditions are used for the cubic spline.

  ### Example. Volatility surface interpolation

  ```python
  dtype = np.float64
  times = tf.constant([2., 2.5, 3, 4.5], dtype=dtype)
  strikes = tf.constant([16, 22, 35], dtype=dtype)

  times_data = tf.constant([1.5, 2.5, 3.5, 4.5, 5.5], dtype=dtype)
  # Corresponding squared volatility values
  sigma_square_data = tf.constant(
      [[0.15, 0.25, 0.35, 0.4, 0.45, 0.4],
       [0.2, 0.35, 0.55, 0.45, 0.4, 0.6],
       [0.3, 0.45, 0.25, 0.4, 0.5, 0.65],
       [0.25, 0.25, 0.45, 0.25, 0.5, 0.55],
       [0.35, 0.35, 0.25, 0.4, 0.55, 0.65]], dtype=dtype)
  # Interpolation is done for the total variance
  total_variance = tf.expand_dims(times_data, -1) * sigma_square_data
  # Corresponding strike values. Notice we need to broadcast to the shape of
  # `sigma_square_data`
  strike_data = tf.broadcast_to(
      tf.constant([15, 25, 35, 40, 50, 55], dtype=dtype), [5, 6])
  # Interpolate total variance on for coordinates `(times, strikes)`
  interpolator = Interpolation2D(times_data, strike_data, total_variance,
                                 dtype=dtype)
  interpolated_values = interpolator.interpolate(times, strikes)
  ```
  """

    def __init__(self, x_data: types.RealTensor, y_data: types.RealTensor, z_data: types.RealTensor, dtype: tf.DType=None, name: str=None):
        if False:
            while True:
                i = 10
        'Initialize the 2d-interpolation object.\n\n    Args:\n      x_data: A `Tensor` of real `dtype` and shape\n        `batch_shape + [num_x_data_points]`.\n        Defines the x-coordinates of the input data. `num_x_data_points` should\n        be >= 2. The elements of `x_data` should be in a non-decreasing order.\n      y_data: A `Tensor` of the same `dtype` as `x_data` and shape\n        `batch_shape + [num_x_data_points, num_y_data_points]`. Defines the\n        y-coordinates of the input data. `num_y_data_points` should be >= 2.\n        The elements of `y_data` should be in a non-decreasing order along last\n        dimension.\n      z_data: A `Tensor` of the same shape and `dtype` as `y_data`. Defines the\n        z-coordinates of the input data (i.e., the function values).\n      dtype: Optional dtype for the input `Tensor`s.\n        Default value: `None` which maps to the default dtype inferred by\n        TensorFlow.\n      name: Python `str` name prefixed to ops created by this class.\n        Default value: `None` which is mapped to the default name\n        `interpolation_2d`.\n    '
        name = name or 'interpolation_2d'
        with tf.name_scope(name):
            self._xdata = tf.convert_to_tensor(x_data, dtype=dtype, name='x_data')
            self._dtype = dtype or self._xdata.dtype
            self._ydata = tf.convert_to_tensor(y_data, dtype=self._dtype, name='y_data')
            self._zdata = tf.convert_to_tensor(z_data, dtype=self._dtype, name='z_data')
            self._name = name
            self._spline_yz = cubic.build_spline(self._ydata, self._zdata, name='spline_y_direction')

    def interpolate(self, x: types.RealTensor, y: types.RealTensor, name: str=None):
        if False:
            print('Hello World!')
        'Performs 2-D interpolation on a specified set of points.\n\n    Args:\n      x: Real-valued `Tensor` of shape `batch_shape + [num_points]`.\n        Defines the x-coordinates at which the interpolation should be\n        performed. Note that `batch_shape` should be the same as in the\n        underlying data.\n      y: A `Tensor` of the same shape and `dtype` as `x`.\n        Defines the y-coordinates at which the interpolation should be\n        performed.\n      name: Python `str` name prefixed to ops created by this function.\n        Default value: `None` which is mapped to the default name\n        `interpolate`.\n\n    Returns:\n      A `Tensor` of the same shape and `dtype` as `x`. Represents the\n      interpolated values of the function on for the coordinates\n      `(x, y)`.\n    '
        name = name or self._name + '_interpolate'
        with tf.name_scope(name):
            x = tf.convert_to_tensor(x, dtype=self._dtype, name='x')
            y = tf.convert_to_tensor(y, dtype=self._dtype, name='y')
            y = tf.expand_dims(y, axis=-2)
            xy = cubic.interpolate(y, self._spline_yz, name='interpolation_in_y_direction')
            xy_rank = xy.shape.rank
            perm = [xy_rank - 1] + list(range(xy_rank - 1))
            yx = tf.transpose(xy, perm=perm)
            perm_original = list(range(1, xy_rank)) + [0]
            x = tf.expand_dims(tf.transpose(x, [xy_rank - 2] + list(range(xy_rank - 2))), axis=-1)
            z_values = linear.interpolate(x, self._xdata, yx)
            return tf.squeeze(tf.transpose(z_values, perm=perm_original), axis=-2)