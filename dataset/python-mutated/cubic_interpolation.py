"""Cubic Spline interpolation framework."""
import enum
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
__all__ = ['BoundaryConditionType', 'SplineParameters', 'build', 'interpolate']

@enum.unique
class BoundaryConditionType(enum.Enum):
    """Specifies which boundary condition type to use for the cubic interpolation.

  * `NATURAL`: the cubic interpolation set second derivative equal to zero
  at boundaries.
  * `CLAMPED`: the cubic interpolation set first derivative equal to zero
  at boundaries.
  * `FIXED_FIRST_DERIVATIVE`: the cubic interpolation set first derivative to
  certain value at boundaries.
  """
    NATURAL = 1
    CLAMPED = 2
    FIXED_FIRST_DERIVATIVE = 3

@tff_utils.dataclass
class SplineParameters:
    """Cubic spline parameters.

  Attributes:
    x_data: A real `Tensor` of shape batch_shape + [num_points] containing
      X-coordinates of the spline.
    y_data: A `Tensor` of the same shape and `dtype` as `x_data` containing
      Y-coordinates of the spline.
    spline_coeffs: A `Tensor` of the same shape and `dtype` as `x_data`
      containing spline interpolation coefficients
  """
    x_data: types.RealTensor
    y_data: types.RealTensor
    spline_coeffs: types.RealTensor

def build(x_data: types.RealTensor, y_data: types.RealTensor, boundary_condition_type: BoundaryConditionType=None, left_boundary_value: types.RealTensor=None, right_boundary_value: types.RealTensor=None, validate_args: bool=False, dtype: tf.DType=None, name=None) -> SplineParameters:
    if False:
        for i in range(10):
            print('nop')
    "Builds a SplineParameters interpolation object.\n\n  Given a `Tensor` of state points `x_data` and corresponding values `y_data`\n  creates an object that contains interpolation coefficients. The object can be\n  used by the `interpolate` function to get interpolated values for a set of\n  state points `x` using the cubic spline interpolation algorithm.\n  It assumes that the second derivative at the first and last spline points\n  are zero. The basic logic is explained in [1] (see also, e.g., [2]).\n\n  Repeated entries in `x_data` are only allowed for the *right* boundary values\n  of `x_data`.\n  For example, `x_data` can be `[1., 2, 3. 4., 4., 4.]` but not\n  `[1., 1., 2., 3.]`. The repeated values play no role in interpolation and are\n  useful only for interpolating multiple splines with different numbers of data\n  point. It is user responsibility to verify that the corresponding\n  values of `y_data` are the same for the repeated values of `x_data`.\n\n  Typical Usage Example:\n\n  ```python\n  import tensorflow as tf\n  import tf_quant_finance as tff\n  import numpy as np\n\n  x_data = tf.linspace(-5.0, 5.0,  num=11)\n  y_data = 1.0/(1.0 + x_data**2)\n  spline = tff.math.interpolation.cubic.build_spline(x_data, y_data)\n  x_args = [3.3, 3.4, 3.9]\n\n  tff.math.interpolation.cubic.interpolate(x_args, spline)\n  # Expected: [0.0833737 , 0.07881707, 0.06149562]\n  ```\n\n  #### References:\n  [1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.\n    Link: https://api.semanticscholar.org/CorpusID:10976311\n  [2]: R. Pienaar, M Choudhry. Fitting the term structure of interest rates:\n    the practical implementation of cubic spline methodology.\n    Link:\n    http://yieldcurve.com/mktresearch/files/PienaarChoudhry_CubicSpline2.pdf\n\n  Args:\n    x_data: A real `Tensor` of shape `[..., num_points]` containing\n      X-coordinates of points to fit the splines to. The values have to be\n      monotonically non-decreasing along the last dimension.\n    y_data: A `Tensor` of the same shape and `dtype` as `x_data` containing\n      Y-coordinates of points to fit the splines to.\n    boundary_condition_type: Boundary condition type for current cubic\n      interpolation. Instance of BoundaryConditionType enum.\n      Default value: `None` which maps to `BoundaryConditionType.NATURAL`.\n    left_boundary_value: Set to non-empty value IFF boundary_condition_type is\n      FIXED_FIRST_DERIVATIVE, in which case set to cubic spline's first\n      derivative at `x_data[..., 0]`.\n    right_boundary_value: Set to non-empty value IFF boundary_condition_type is\n      FIXED_FIRST_DERIVATIVE, in which case set to cubic spline's first\n      derivative at `x_data[..., num_points - 1]`.\n    validate_args: Python `bool`. When `True`, verifies if elements of `x_data`\n      are sorted in the last dimension in non-decreasing order despite possibly\n      degrading runtime performance.\n      Default value: False.\n    dtype: Optional dtype for both `x_data` and `y_data`.\n      Default value: `None` which maps to the default dtype inferred by\n        TensorFlow.\n    name: Python `str` name prefixed to ops created by this function.\n      Default value: `None` which is mapped to the default name\n        `cubic_spline_build`.\n\n  Returns:\n    An instance of `SplineParameters`.\n  "
    if boundary_condition_type is None:
        boundary_condition_type = BoundaryConditionType.NATURAL
    if name is None:
        name = 'cubic_spline_build'
    with tf.name_scope(name):
        x_data = tf.convert_to_tensor(x_data, dtype=dtype, name='x_data')
        y_data = tf.convert_to_tensor(y_data, dtype=dtype, name='y_data')
        if validate_args:
            assert_sanity_check = [_validate_arguments(x_data)]
        else:
            assert_sanity_check = []
        (x_data, y_data) = tff_utils.broadcast_common_batch_shape(x_data, y_data)
        if boundary_condition_type == BoundaryConditionType.FIXED_FIRST_DERIVATIVE:
            if left_boundary_value is None or right_boundary_value is None:
                raise ValueError('Expected non-empty left_boundary_value/right_boundary_value when boundary_condition_type is FIXED_FIRST_DERIVATIVE, actual left_boundary_value {0}, actual right_boundary_value {1}'.format(left_boundary_value, right_boundary_value))
        with tf.compat.v1.control_dependencies(assert_sanity_check):
            spline_coeffs = _calculate_spline_coeffs(x_data, y_data, boundary_condition_type, left_boundary_value, right_boundary_value)
        return SplineParameters(x_data=x_data, y_data=y_data, spline_coeffs=spline_coeffs)

def interpolate(x: types.RealTensor, spline_data: SplineParameters, optimize_for_tpu: bool=False, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        for i in range(10):
            print('nop')
    'Interpolates spline values for the given `x` and the `spline_data`.\n\n  Constant extrapolation is performed for the values outside the domain\n  `spline_data.x_data`. This means that for `x > max(spline_data.x_data)`,\n  `interpolate(x, spline_data) = spline_data.y_data[-1]`\n  and for  `x < min(spline_data.x_data)`,\n  `interpolate(x, spline_data) = spline_data.y_data[0]`.\n\n  For the interpolation formula refer to p.548 of [1].\n\n  #### References:\n  [1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.\n    Link: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf\n\n  Args:\n    x: A real `Tensor` of shape `batch_shape + [num_points]`.\n    spline_data: An instance of `SplineParameters`. `spline_data.x_data` should\n      have the same batch shape as `x`.\n    optimize_for_tpu: A Python bool. If `True`, the algorithm uses one-hot\n      encoding to lookup indices of `x` in `spline_data.x_data`. This\n      significantly improves performance of the algorithm on a TPU device but\n      may slow down performance on the CPU.\n      Default value: `False`.\n    dtype: Optional dtype for `x`.\n      Default value: `None` which maps to the default dtype inferred by\n        TensorFlow.\n    name: Python `str` name prefixed to ops created by this function.\n      Default value: `None` which is mapped to the default name\n        `cubic_spline_interpolate`.\n\n  Returns:\n      A `Tensor` of the same shape and `dtype` as `x`. Represents\n      the interpolated values.\n\n  Raises:\n    ValueError:\n      If `x` batch shape is different from `spline_data.x_data` batch\n      shape.\n  '
    name = name or 'cubic_spline_interpolate'
    with tf.name_scope(name):
        x = tf.convert_to_tensor(x, dtype=dtype, name='x')
        dtype = x.dtype
        x_data = spline_data.x_data
        y_data = spline_data.y_data
        spline_coeffs = spline_data.spline_coeffs
        (x, x_data, y_data, spline_coeffs) = tff_utils.broadcast_common_batch_shape(x, x_data, y_data, spline_coeffs)
        indices = tf.searchsorted(x_data, x, side='right') - 1
        lower_encoding = tf.maximum(indices, 0)
        upper_encoding = tf.minimum(indices + 1, tff_utils.get_shape(x_data)[-1] - 1)
        if optimize_for_tpu:
            x_data_size = tff_utils.get_shape(x_data)[-1]
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
        x0 = get_slice(x_data, lower_encoding)
        x1 = get_slice(x_data, upper_encoding)
        dx = x1 - x0
        y0 = get_slice(y_data, lower_encoding)
        y1 = get_slice(y_data, upper_encoding)
        dy = y1 - y0
        spline_coeffs0 = get_slice(spline_coeffs, lower_encoding)
        spline_coeffs1 = get_slice(spline_coeffs, upper_encoding)
        t = (x - x0) / dx
        t = tf.where(dx > 0, t, tf.zeros_like(t))
        df = (t + 1.0) * spline_coeffs1 * 2.0 - (t - 2.0) * spline_coeffs0 * 2.0
        df1 = df * t * (t - 1) / 6.0
        result = y0 + t * dy + dx * dx * df1
        upper_bound = tf.expand_dims(tf.reduce_max(x_data, -1), -1) + tf.zeros_like(result)
        lower_bound = tf.expand_dims(tf.reduce_min(x_data, -1), -1) + tf.zeros_like(result)
        result = tf.where(tf.logical_and(x <= upper_bound, x >= lower_bound), result, tf.where(x > upper_bound, y0, y1))
        return result

def _calculate_spline_coeffs_natural(dx, superdiag, subdiag, diag_values, rhs, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Calculates spline coefficients for the NATURAL boundary condition.'
    corr_term = tf.logical_or(tf.equal(superdiag, 0), tf.equal(subdiag, 0))
    diag_values_corr = tf.where(corr_term, tf.ones_like(diag_values), diag_values)
    superdiag_corr = tf.where(tf.equal(subdiag, 0), tf.zeros_like(superdiag), superdiag)
    subdiag_corr = tf.where(tf.equal(superdiag, 0), tf.zeros_like(subdiag), subdiag)
    diagonals = tf.stack([superdiag_corr, diag_values_corr, subdiag_corr], axis=-2)
    rhs = tf.where(corr_term, tf.zeros_like(rhs), rhs)
    spline_coeffs = tf.linalg.tridiagonal_solve(diagonals, rhs, partial_pivoting=False)
    zero = tf.zeros_like(dx[..., :1], dtype=dtype)
    spline_coeffs = tf.concat([zero, spline_coeffs, zero], axis=-1)
    return spline_coeffs

def _calculate_spline_coeffs_clamped_or_first_derivative(dx, dd, superdiag, subdiag, diag_values, rhs, dtype, boundary_condition_type, left_boundary_value=None, right_boundary_value=None):
    if False:
        while True:
            i = 10
    'Calculates the coefficients for the spline interpolation if the boundary condition type is CLAMPED/FIXED_FIRST_DERIVATIVE.'
    zero = tf.zeros_like(dx[..., :1], dtype=dtype)
    one = tf.ones_like(dx[..., :1], dtype=dtype)
    diag_values = tf.concat([2.0 * dx[..., :1], diag_values, zero], axis=-1)
    superdiag = tf.concat([dx[..., :1], superdiag, zero], axis=-1)
    subdiag = tf.concat([zero, subdiag, dx[..., -1:]], axis=-1)
    dx = tf.concat((one, dx, zero), axis=-1)
    dx_right = dx[..., 1:]
    dx_left = dx[..., :-1]
    right_boundary = tf.math.logical_and(tf.equal(dx_right, 0), tf.not_equal(dx_left, 0))
    diag_values = tf.where(right_boundary, 2.0 * dx_left, diag_values)
    diag_values = tf.where(tf.equal(dx_left, 0), one, diag_values)
    diagonals = tf.stack([superdiag, diag_values, subdiag], axis=-2)
    left_boundary_tensor = tf.zeros_like(dx[..., :1], dtype=dtype)
    right_boundary_tensor = tf.zeros_like(dx[..., :1], dtype=dtype)
    if boundary_condition_type == BoundaryConditionType.FIXED_FIRST_DERIVATIVE:
        left_boundary_tensor = tf.convert_to_tensor(left_boundary_value, dtype=dtype, name='left_boundary_value')
        right_boundary_tensor = tf.convert_to_tensor(right_boundary_value, dtype=dtype, name='right_boundary_value')
    top_rhs = 3.0 * (dd[..., :1] - left_boundary_tensor[..., :1])
    rhs = tf.concat([top_rhs, rhs, zero], axis=-1)
    dd_left = tf.concat((one, dd), axis=-1)
    bottom_rhs = -3.0 * (dd_left - right_boundary_tensor[..., :1])
    rhs = tf.where(right_boundary, bottom_rhs, rhs)
    rhs = tf.where(tf.equal(dd_left, 0), zero, rhs)
    spline_coeffs = tf.linalg.tridiagonal_solve(diagonals, rhs, partial_pivoting=False)
    return spline_coeffs

def _calculate_spline_coeffs(x_data, y_data, boundary_condition_type=BoundaryConditionType.NATURAL, left_boundary_value=None, right_boundary_value=None):
    if False:
        i = 10
        return i + 15
    "Calculates the coefficients for the spline interpolation.\n\n  These are the values of the second derivative of the spline at `x_data`.\n  See p.548 of [1].\n\n  #### Below formula is for natural condition type.\n  It is an outline of the function when number of observations if equal to 7.\n  The coefficients are obtained by building and solving a tridiagonal linear\n  system of equations with symmetric matrix\n   1,  0,  0,    0,    0,   0,  0\n  dx0  w0, dx1,  0,    0,   0,  0\n   0, dx1,  w1,  dx2,  0,   0,  0\n   0,  0,  dx2,  w2,  dx3,  0,  0\n   0,  0,   0,   dx3,  w3, dx4, 0\n   0,  0,   0,   0,   dx4,  w4, dx5\n   0,  0,   0,   0,    0,   0,  1\n   where:\n   dxn = x_data[n+1] - x_data[n]\n   wn = 2 * (dx[n] + dx[n+1])\n\n   and the right hand side of the equation is:\n   [[0],\n    [3*( (y2-y1)/dx1 - (y1-y0)/dx0],\n    [3*( (y3-y2)/dx2 - (y2-y1)/dx1],\n    [3*( (y4-y3)/dx3 - (y3-y2)/dx2],\n    [3*( (y5-y4)/dx4 - (y4-y3)/dx3],\n    [3*( (y6-y5)/dx5 - (y5-y4)/dx4],\n    [0]\n   ]\n\n   with yi = y_data[..., i]\n\n   Solve for `spline_coeffs`, so that  matrix * spline_coeffs = rhs\n\n   the solution is the `spline_coeffs` parameter of the spline equation:\n\n   y_pred = a(spline_coeffs) * t^3 + b(spline_coeffs) * t^2\n            + c(spline_coeffs) * t + d(spline_coeffs)\n   with t being the proportion of the difference between the x value of\n   the spline used and the nx_value of the next spline:\n\n   t = (x - x_data[:,n]) / (x_data[:,n+1]-x_data[:,n])\n\n   and `a`, `b`, `c`, and `d` are functions of `spline_coeffs` and `x_data` and\n   are provided in the `interpolate` function.\n\n  #### Below formula is for clamped/first_derivative condition type.\n  Similar to natural condition type, let us assume the number of observations\n  is equal to 7. The underlying mathematics can be found in [2].\n  left hand side matrix:\n  2*dx0, dx0,  0,   0,    0,   0,  0\n   dx0    w0, dx1,  0,    0,   0,  0\n    0,   dx1,  w1,  dx2,  0,   0,  0\n    0,    0,  dx2,  w2,  dx3,  0,  0\n    0,    0,   0,   dx3,  w3, dx4, 0\n    0,    0,   0,   0,   dx4,  w4, dx5\n    0,    0,   0,   0,    0,  dx5, 2*dx5\n   where:\n   dxn and wn is same as natural contition case.\n\n   and the right hand side of the equation is:\n   [[3* ((y1-y0)/dx0 - lb)],\n    [3*( (y2-y1)/dx1 - (y1-y0)/dx0],\n    [3*( (y3-y2)/dx2 - (y2-y1)/dx1],\n    [3*( (y4-y3)/dx3 - (y3-y2)/dx2],\n    [3*( (y5-y4)/dx4 - (y4-y3)/dx3],\n    [3*( (y6-y5)/dx5 - (y5-y4)/dx4],\n    [-3*((y6-y5)/dx5 - rb)]\n   ]\n   where dxn, yi is same as natural case.\n   lb is specified first derivative at left boundary.\n   rb is specified first derivative at right boundary.\n\n  #### Special handling for right padding, imagine the number of observations\n  is equal to 7. While there are 2 repeated points as right padding.\n  The left hand matrix needs to be:\n  2*dx0, dx0,  0,   0,    0,   0,  0     0,  0\n   dx0    w0, dx1,  0,    0,   0,  0     0,  0\n    0,   dx1,  w1,  dx2,  0,   0,  0     0,  0\n    0,    0,  dx2,  w2,  dx3,  0,  0     0,  0\n    0,    0,   0,   dx3,  w3, dx4, 0     0,  0\n    0,    0,   0,   0,   dx4,  w4, dx5   0,  0\n    0,    0,   0,   0,    0,  dx5, 2*dx5 0,  0\n    0,    0,   0,   0,    0,  0,   0,    1,  0\n    0,    0,   0,   0,    0,  0,   0,    0,  1\n\n   The right hand matrix needs to be:\n    [[3* ((y1-y0)/dx0 - lb)],\n    [3*( (y2-y1)/dx1 - (y1-y0)/dx0],\n    [3*( (y3-y2)/dx2 - (y2-y1)/dx1],\n    [3*( (y4-y3)/dx3 - (y3-y2)/dx2],\n    [3*( (y5-y4)/dx4 - (y4-y3)/dx3],\n    [3*( (y6-y5)/dx5 - (y5-y4)/dx4],\n    [-3*((y6-y5)/dx5 - rb)],\n    [0],\n    [0]\n   ]\n\n  #### References:\n  [1]: R. Sedgewick, Algorithms in C, 1990, p. 545-550.\n    Link: http://index-of.co.uk/Algorithms/Algorithms%20in%20C.pdf\n\n  Args:\n    x_data: A real `Tensor` of shape `[..., num_points]` containing\n      X-coordinates of points to fit the splines to. The values have to be\n      monotonically non-decreasing along the last dimension.\n    y_data: A `Tensor` of the same shape and `dtype` as `x_data` containing\n      Y-coordinates of points to fit the splines to.\n    boundary_condition_type: Boundary condition type for current cubic\n      interpolation.\n    left_boundary_value: Set to non-empty value IFF boundary_condition_type is\n      FIXED_FIRST_DERIVATIVE, in which case set to cubic spline's\n      first derivative at x_data[: 0].\n    right_boundary_value: Set to non-empty value IFF boundary_condition_type is\n      FIXED_FIRST_DERIVATIVE, in which case set to cubic spline's\n      first derivative at x_data[: num_points - 1]\n\n  Returns:\n     A `Tensor` of the same shape and `dtype` as `x_data`. Represents the\n     spline coefficients for the cubic spline interpolation.\n  [2]: http://macs.citadel.edu/chenm/343.dir/09.dir/lect3_4.pdf\n  "
    dx = x_data[..., 1:] - x_data[..., :-1]
    dd = (y_data[..., 1:] - y_data[..., :-1]) / dx
    dd = tf.where(tf.equal(dx, 0), tf.zeros_like(dd), dd)
    rhs = -3 * (dd[..., :-1] - dd[..., 1:])
    diag_values = 2.0 * (x_data[..., 2:] - x_data[..., :-2])
    superdiag = dx[..., 1:]
    subdiag = dx[..., :-1]
    if boundary_condition_type == BoundaryConditionType.NATURAL:
        return _calculate_spline_coeffs_natural(dx, superdiag, subdiag, diag_values, rhs, x_data.dtype)
    elif boundary_condition_type in [BoundaryConditionType.FIXED_FIRST_DERIVATIVE, BoundaryConditionType.CLAMPED]:
        return _calculate_spline_coeffs_clamped_or_first_derivative(dx, dd, superdiag, subdiag, diag_values, rhs, x_data.dtype, boundary_condition_type, left_boundary_value, right_boundary_value)

def _validate_arguments(x_data):
    if False:
        print('Hello World!')
    'Checks that input arguments are in the non-decreasing order.'
    diffs = x_data[..., 1:] - x_data[..., :-1]
    return tf.compat.v1.debugging.assert_greater_equal(diffs, tf.zeros_like(diffs), message='x_data is not sorted in non-decreasing order.')