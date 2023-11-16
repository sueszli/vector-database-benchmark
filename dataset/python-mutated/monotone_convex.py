"""The monotone convex interpolation method.

The monotone convex method is a scheme devised by Hagan and West (Ref [1]). It
is a commonly used method to interpolate interest rate yield curves. For
more details see Refs [1, 2].

It is important to point out that the monotone convex method *does not* solve
the standard interpolation problem but a modified one as described below.

Suppose we are given a strictly increasing sequence of scalars (which we will
refer to as time) `[t_1, t_2, ... t_n]` and a set of values
`[f_1, f_2, ... f_n]`.
The aim is to find a function `f(t)` defined on the interval `[0, t_n]` which
satisfies (in addition to continuity and positivity conditions, see Section 6
of Ref [2] for more details) the following

```
  Integral[f(u), t_{i-1} <= u <= t_i] = f_i,  with t_0 = 0

```

In the context of interest rate curve building, `f(t)` corresponds to the
instantaneous forward rate at time `t` and the `f_i` correspond to the
discrete forward rates that apply to the time period `[t_{i-1}, t_i]`.

This implementation of the method currently supports batching along the
interpolation times but not along the interpolated curves (i.e. it is possible
to evaluate the `f(t)` for `t` as a vector of times but not build multiple
curves at the same time).


#### References:

[1]: Patrick Hagan & Graeme West. Interpolation Methods for Curve Construction.
  Applied Mathematical Finance. Vol 13, No. 2, pp 89-129. June 2006.
  https://www.researchgate.net/publication/24071726_Interpolation_Methods_for_Curve_Construction
[2]: Patrick Hagan & Graeme West. Methods for Constructing a Yield Curve.
  Wilmott Magazine, pp. 70-81. May 2008.
"""
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.math import piecewise
from tf_quant_finance.math.diff_ops import diff
from tf_quant_finance.rates.analytics import forwards

def interpolate(times: types.RealTensor, interval_values: types.RealTensor, interval_times: types.RealTensor, validate_args: bool=False, dtype: tf.DType=None, name: str=None):
    if False:
        print('Hello World!')
    "Performs the monotone convex interpolation.\n\n  The monotone convex method is a scheme devised by Hagan and West (Ref [1]). It\n  is a commonly used method to interpolate interest rate yield curves. For\n  more details see Refs [1, 2].\n\n  It is important to point out that the monotone convex method *does not* solve\n  the standard interpolation problem but a modified one as described below.\n\n  Suppose we are given a strictly increasing sequence of scalars (which we will\n  refer to as time) `[t_1, t_2, ... t_n]` and a set of values\n  `[f_1, f_2, ... f_n]`.\n  The aim is to find a function `f(t)` defined on the interval `[0, t_n]` which\n  satisfies (in addition to continuity and positivity conditions) the following\n\n  ```None\n    Integral[f(u), t_{i-1} <= u <= t_i] = f_i,  with t_0 = 0\n\n  ```\n\n  In the context of interest rate curve building, `f(t)` corresponds to the\n  instantaneous forward rate at time `t` and the `f_i` correspond to the\n  discrete forward rates that apply to the time period `[t_{i-1}, t_i]`.\n  Furthermore, the integral of the forward curve is related to the yield curve\n  by\n\n  ```None\n    Integral[f(u), 0 <= u <= t] = r(t) * t\n\n  ```\n\n  where `r(t)` is the interest rate that applies between `[0, t]` (the yield of\n  a zero coupon bond paying a unit of currency at time `t`).\n\n  This function computes both the interpolated value and the integral along\n  the segment containing the supplied time. Specifically, given a time `t` such\n  that `t_k <= t <= t_{k+1}`, this function computes the interpolated value\n  `f(t)` and the value `Integral[f(u), t_k <= u <= t]`.\n\n  This implementation of the method currently supports batching along the\n  interpolation times but not along the interpolated curves (i.e. it is possible\n  to evaluate the `f(t)` for `t` as a vector of times but not build multiple\n  curves at the same time).\n\n  #### Example\n\n  ```python\n  interval_times = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0], dtype=dtype)\n  interval_values = tf.constant([0.05, 0.051, 0.052, 0.053, 0.055],\n                                dtype=dtype)\n  times = tf.constant([0.25, 0.5, 1.0, 2.0, 3.0, 1.1], dtype=dtype)\n  # Returns the following two values:\n  # interpolated = [0.0505, 0.05133333, 0.05233333, 0.054, 0.0555, 0.05241]\n  # integrated =  [0, 0, 0, 0, 0.055, 0.005237]\n  # Note that the first four integrated values are zero. This is because\n  # those interpolation time are at the start of their containing interval.\n  # The fourth value (i.e. at 3.0) is not zero because this is the last\n  # interval (i.e. it is the integral from 2.0 to 3.0).\n  interpolated, integrated = interpolate(\n      times, interval_values, interval_times)\n  ```\n\n  #### References:\n\n  [1]: Patrick Hagan & Graeme West. Interpolation Methods for Curve\n    Construction. Applied Mathematical Finance. Vol 13, No. 2, pp 89-129.\n    June 2006.\n    https://www.researchgate.net/publication/24071726_Interpolation_Methods_for_Curve_Construction\n  [2]: Patrick Hagan & Graeme West. Methods for Constructing a Yield Curve.\n    Wilmott Magazine, pp. 70-81. May 2008.\n\n  Args:\n    times: Non-negative rank 1 `Tensor` of any size. The times for which the\n      interpolation has to be performed.\n    interval_values: Rank 1 `Tensor` of the same shape and dtype as\n      `interval_times`. The values associated to each of the intervals specified\n      by the `interval_times`. Must have size at least 2.\n    interval_times: Strictly positive rank 1 `Tensor` of real dtype containing\n      increasing values. The endpoints of the intervals (i.e. `t_i` above.).\n      Note that the left end point of the first interval is implicitly assumed\n      to be 0. Must have size at least 2.\n    validate_args: Python bool. If true, adds control dependencies to check that\n      the `times` are bounded by the `interval_endpoints`.\n      Default value: False\n    dtype: `tf.Dtype` to use when converting arguments to `Tensor`s. If not\n      supplied, the default Tensorflow conversion will take place. Note that\n      this argument does not do any casting.\n      Default value: None.\n    name: Python `str` name prefixed to Ops created by this class.\n      Default value: None which is mapped to the default name 'interpolation'.\n\n  Returns:\n    A 2-tuple containing\n      interpolated_values: Rank 1 `Tensor` of the same size and dtype as the\n        `times`. The interpolated values at the supplied times.\n      integrated_values: Rank 1 `Tensor` of the same size and dtype as the\n        `times`. The integral of the interpolated function. The integral is\n        computed from the largest interval time that is smaller than the time\n        up to the given time.\n  "
    if name is None:
        name = 'interpolate'
    with tf.name_scope(name):
        times = tf.convert_to_tensor(times, dtype=dtype, name='times')
        interval_times = tf.convert_to_tensor(interval_times, dtype=dtype, name='interval_times')
        interval_values = tf.convert_to_tensor(interval_values, dtype=dtype, name='interval_values')
        control_deps = []
        if validate_args:
            control_deps = [tf.compat.v1.debugging.assert_non_negative(times), tf.compat.v1.debugging.assert_positive(interval_times)]
        with tf.compat.v1.control_dependencies(control_deps):
            endpoint_values = _interpolate_adjacent(interval_times, interval_values)
            endpoint_times = tf.concat([[0.0], interval_times], axis=0)
            intervals = piecewise.find_interval_index(times, endpoint_times, last_interval_is_closed=True)
            f_left = tf.gather(endpoint_values, intervals)
            f_right = tf.gather(endpoint_values, intervals + 1)
            fd = tf.gather(interval_values, intervals)
            t_left = tf.gather(endpoint_times, intervals)
            t_right = tf.gather(endpoint_times, intervals + 1)
            interval_lengths = t_right - t_left
            x = (times - t_left) / interval_lengths
            g0 = f_left - fd
            g1 = f_right - fd
            g1plus2g0 = g1 + 2 * g0
            g0plus2g1 = g0 + 2 * g1
            result = tf.zeros_like(times)
            integrated = tf.zeros_like(times)
            (is_region_1, region_1_value, integrated_value_1) = _region_1(g1plus2g0, g0plus2g1, g0, g1, x)
            result = tf.where(is_region_1, region_1_value, result)
            integrated = tf.where(is_region_1, integrated_value_1, integrated)
            (is_region_2, region_2_value, integrated_value_2) = _region_2(g1plus2g0, g0plus2g1, g0, g1, x)
            result = tf.where(is_region_2, region_2_value, result)
            integrated = tf.where(is_region_2, integrated_value_2, integrated)
            (is_region_3, region_3_value, integrated_value_3) = _region_3(g1plus2g0, g0plus2g1, g0, g1, x)
            result = tf.where(is_region_3, region_3_value, result)
            integrated = tf.where(is_region_3, integrated_value_3, integrated)
            (is_region_4, region_4_value, integrated_value_4) = _region_4(g1plus2g0, g0plus2g1, g0, g1, x)
            result = tf.where(is_region_4, region_4_value, result)
            integrated = tf.where(is_region_4, integrated_value_4, integrated)
            g0_eps = tf.abs(tf.math.nextafter(fd, f_left) - fd) * 1.1
            g1_eps = tf.abs(tf.math.nextafter(fd, f_right) - fd) * 1.1
            is_origin = (tf.abs(g0) <= g0_eps) & (tf.abs(g1) <= g1_eps)
            result = tf.where(is_origin, tf.zeros_like(result), result)
            integrated = tf.where(is_origin, tf.zeros_like(integrated), integrated)
            return (result + fd, (integrated + fd * x) * interval_lengths)

def interpolate_forward_rate(interpolation_times, reference_times, yields=None, discrete_forwards=None, validate_args=False, dtype=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Interpolates instantaneous forward rate to supplied times .\n\n    Applies the Hagan West procedure to interpolate either a zero coupon yield\n    curve or a discrete forward curve to a given set of times to compute\n    the instantaneous forward rate for those times.\n\n    A zero coupon yield curve is specified by a set\n    of times and the yields on zero coupon bonds expiring at those\n    times. A discrete forward rate curve specifies the interest rate that\n    applies between two times in the future as seen from the current time.\n    The relation between the two sets of curve is as follows. Suppose the\n    yields on zero coupon bonds expiring at times `[t_1, ..., t_n]` are\n    `[r_1, ..., r_n]`, then the forward rate between time `[t_i, t_{i+1}]` is\n    denoted `f(0; t_i, t_{i+1})` and given by\n\n    ```None\n      f(0; t_i, t_{i+1}) = (r_{i+1} t_{i+1} - r_i t_i) / (t_{i+1} - t_i)\n    ```\n    This function uses the Hagan West algorithm to perform the interpolation.\n    This scheme interpolates on the forward curve. If `yields` are specified\n    instead of `discrete_forwards` then they are first converted to the\n    discrete forwards before interpolation.\n    For more details on the interpolation procedure, see Ref. [1].\n\n  #### Example\n\n  ```python\n    dtype = np.float64\n    # Market data.\n    reference_times = np.array([1.0, 2.0, 3.0, 4,0, 5.0], dtype=dtype)\n    yields = np.array([2.75, 4.0, 4.75, 5.0, 4.75], dtype=dtype) / 100\n\n    # Times for which the interpolated values are required.\n    interpolation_times = np.array([0.3, 1.3, 2.1, 4.5], dtype=dtype)\n    interpolated_forwards = interpolate_forward_rates(\n        interpolation_times,\n        reference_times=reference_times,\n        yields=yields)\n\n    # Produces: [0.0229375, 0.05010625, 0.0609, 0.03625].\n  ```\n\n  #### References:\n\n  [1]: Patrick Hagan & Graeme West. Methods for Constructing a Yield Curve.\n    Wilmott Magazine, pp. 70-81. May 2008.\n\n  Args:\n    interpolation_times: Non-negative rank 1 `Tensor` of any size. The times for\n      which the interpolation has to be performed.\n    reference_times: Strictly positive rank 1 `Tensor` of real dtype containing\n      increasing values. The expiry times of the underlying zero coupon bonds.\n    yields: Optional rank 1 `Tensor` of the same shape and dtype as\n      `reference_times`, if supplied. The yield rate of zero coupon bonds\n      expiring at the corresponding time in the `reference_times`. Either this\n      argument or the `discrete_forwards` must be supplied (but not both).\n      Default value: None.\n    discrete_forwards: Optional rank 1 `Tensor` of the same shape and dtype as\n      `reference_times`, if supplied. The `i`th component of the `Tensor` is the\n      forward rate that applies between `reference_times[i-1]` and\n      `reference_times[i]` for `i>0` and between time `0` and\n      `reference_times[0]` for `i=0`. Either this argument or the `yields` must\n      be specified (but not both).\n      Default value: None.\n    validate_args: Python bool. If true, adds control dependencies to check that\n      the `times` are bounded by the `reference_times`.\n      Default value: False\n    dtype: `tf.Dtype` to use when converting arguments to `Tensor`s. If not\n      supplied, the default Tensorflow conversion will take place. Note that\n      this argument does not do any casting.\n      Default value: None.\n    name: Python `str` name prefixed to Ops created by this class.\n      Default value: None which is mapped to the default name\n        'interpolate_forward_rate'.\n\n  Returns:\n      interpolated_forwards: Rank 1 `Tensor` of the same size and dtype as the\n        `interpolation_times`. The interpolated instantaneous forwards at the\n        `interpolation_times`.\n\n  Raises:\n    ValueError if neither `yields` nor `discrete_forwards` are specified or if\n    both are specified.\n  "
    if (yields is None) == (discrete_forwards is None):
        raise ValueError('Exactly one of yields or discrete forwards must be supplied.')
    with tf.compat.v1.name_scope(name, default_name='interpolate_forward_rate', values=[interpolation_times, reference_times, yields, discrete_forwards]):
        if discrete_forwards is not None:
            discrete_forwards = tf.convert_to_tensor(discrete_forwards, dtype=dtype)
        reference_times = tf.convert_to_tensor(reference_times, dtype=dtype)
        interpolation_times = tf.convert_to_tensor(interpolation_times, dtype=dtype)
        if yields is not None:
            yields = tf.convert_to_tensor(yields, dtype=dtype)
            discrete_forwards = forwards.forward_rates_from_yields(yields, reference_times, dtype=dtype)
        (interpolated_forwards, _) = interpolate(interpolation_times, discrete_forwards, reference_times, validate_args=validate_args, dtype=dtype)
        return interpolated_forwards

def interpolate_yields(interpolation_times, reference_times, yields=None, discrete_forwards=None, validate_args=False, dtype=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Interpolates the yield curve to the supplied times.\n\n    Applies the Hagan West procedure to interpolate either a zero coupon yield\n    curve or a discrete forward curve to a given set of times.\n    A zero coupon yield curve is specified by a set\n    of times and the yields on zero coupon bonds expiring at those\n    times. A discrete forward rate curve specifies the interest rate that\n    applies between two times in the future as seen from the current time.\n    The relation between the two sets of curve is as follows. Suppose the\n    yields on zero coupon bonds expiring at times `[t_1, ..., t_n]` are\n    `[r_1, ..., r_n]`, then the forward rate between time `[t_i, t_{i+1}]` is\n    denoted `f(0; t_i, t_{i+1})` and given by\n\n    ```None\n      f(0; t_i, t_{i+1}) = (r_{i+1} t_{i+1} - r_i t_i) / (t_{i+1} - t_i)\n    ```\n\n    This function uses the Hagan West algorithm to perform the interpolation.\n    The interpolation proceeds in two steps. Firstly the discrete forward\n    curve is bootstrapped and an instantaneous forward curve is built. From the\n    instantaneous forward curve, the interpolated yield values are inferred\n    using the relation:\n\n    ```None\n      r(t) = (1/t) * Integrate[ f(s), 0 <= s <= t]\n    ```\n\n    The above equation connects the instantaneous forward curve `f(t)` to the\n    yield curve `r(t)`. The Hagan West procedure uses the Monotone Convex\n    interpolation to create a continuous forward curve. This is then integrated\n    to compute the implied yield rate.\n\n    For more details on the interpolation procedure, see Ref. [1].\n\n  #### Example\n\n  ```python\n    dtype = np.float64\n    reference_times = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)\n    yields = np.array([5.0, 4.75, 4.53333333, 4.775], dtype=dtype)\n\n    # Times for which the interpolated values are required.\n    interpolation_times = np.array([0.25, 0.5, 1.0, 2.0], dtype=dtype)\n\n    interpolated = interpolate_yields(\n        interpolation_times, reference_times, yields=yields)\n    # Produces [5.1171875, 5.09375, 5.0, 4.75]\n  ```\n\n  #### References:\n\n  [1]: Patrick Hagan & Graeme West. Methods for Constructing a Yield Curve.\n    Wilmott Magazine, pp. 70-81. May 2008.\n    https://www.researchgate.net/profile/Patrick_Hagan3/publication/228463045_Methods_for_constructing_a_yield_curve/links/54db8cda0cf23fe133ad4d01.pdf\n\n  Args:\n    interpolation_times: Non-negative rank 1 `Tensor` of any size. The times for\n      which the interpolation has to be performed.\n    reference_times: Strictly positive rank 1 `Tensor` of real dtype containing\n      increasing values. The expiry times of the underlying zero coupon bonds.\n    yields: Optional rank 1 `Tensor` of the same shape and dtype as\n      `reference_times`, if supplied. The yield rate of zero coupon bonds\n      expiring at the corresponding time in the `reference_times`. Either this\n      argument or the `discrete_forwards` must be supplied (but not both).\n      Default value: None.\n    discrete_forwards: Optional rank 1 `Tensor` of the same shape and dtype as\n      `reference_times`, if supplied. The `i`th component of the `Tensor` is the\n      forward rate that applies between `reference_times[i-1]` and\n      `reference_times[i]` for `i>0` and between time `0` and\n      `reference_times[0]` for `i=0`. Either this argument or the `yields` must\n      be specified (but not both).\n      Default value: None.\n    validate_args: Python bool. If true, adds control dependencies to check that\n      the `times` are bounded by the `reference_times`.\n      Default value: False\n    dtype: `tf.Dtype` to use when converting arguments to `Tensor`s. If not\n      supplied, the default Tensorflow conversion will take place. Note that\n      this argument does not do any casting.\n      Default value: None.\n    name: Python `str` name prefixed to Ops created by this class.\n      Default value: None which is mapped to the default name\n        'interpolate_forward_rate'.\n\n  Returns:\n      interpolated_forwards: Rank 1 `Tensor` of the same size and dtype as the\n        `interpolation_times`. The interpolated instantaneous forwards at the\n        `interpolation_times`.\n\n  Raises:\n    ValueError if neither `yields` nor `discrete_forwards` are specified or if\n    both are specified.\n  "
    if (yields is None) == (discrete_forwards is None):
        raise ValueError('Exactly one of yields or discrete forwards must be supplied.')
    with tf.compat.v1.name_scope(name, default_name='interpolate_forward_rate', values=[interpolation_times, reference_times, yields, discrete_forwards]):
        if discrete_forwards is not None:
            discrete_forwards = tf.convert_to_tensor(discrete_forwards, dtype=dtype)
            reference_yields = forwards.yields_from_forward_rates(discrete_forwards, reference_times, dtype=dtype)
        reference_times = tf.convert_to_tensor(reference_times, dtype=dtype)
        interpolation_times = tf.convert_to_tensor(interpolation_times, dtype=dtype)
        if yields is not None:
            reference_yields = tf.convert_to_tensor(yields, dtype=dtype)
            discrete_forwards = forwards.forward_rates_from_yields(reference_yields, reference_times, dtype=dtype)
        (_, integrated_adjustments) = interpolate(interpolation_times, discrete_forwards, reference_times, validate_args=validate_args, dtype=dtype)
        extended_times = tf.concat([[0.0], reference_times], axis=0)
        extended_yields = tf.concat([[0.0], reference_yields], axis=0)
        intervals = piecewise.find_interval_index(interpolation_times, extended_times, last_interval_is_closed=True)
        base_values = tf.gather(extended_yields * extended_times, intervals)
        interpolated = tf.math.divide_no_nan(base_values + integrated_adjustments, interpolation_times)
        return interpolated

def _interpolate_adjacent(times, values, name=None):
    if False:
        return 10
    "Interpolates linearly between adjacent values.\n\n  Suppose `times` are `[t_1, t_2, ..., t_n]` an array of length `n` and\n  values are `[f_1, ... f_n]` of length `n`. This function associates\n  each of the values to the midpoint of the interval i.e. `f_i` is associated\n  to the midpoint of the interval `[t_i, t_{i+1}]`. Then it calculates the\n  values at the interval boundaries by linearly interpolating between adjacent\n  intervals. The first interval is considered to be `[0, t_1]`. The values at\n  the endpoints (i.e. result[0] and result[n]) are computed as follows:\n  `result[0] = values[0] - 0.5 * (result[1] - values[0])` and\n  `result[n] = values[n-1] - 0.5 * (result[n-1] - values[n-1])`.\n  The rationale for these specific values is discussed in Ref. [1].\n\n  Args:\n    times: A rank 1 `Tensor` of real dtype. The times at which the interpolated\n      values are to be computed. The values in the array should be positive and\n      monotonically increasing.\n    values: A rank 1 `Tensor` of the same dtype and shape as `times`. The values\n      assigned to the midpoints of the time intervals.\n    name: Python `str` name prefixed to Ops created by this class.\n      Default value: None which is mapped to the default name\n        'interpolate_adjacent'.\n\n  Returns:\n    interval_values: The values interpolated from the supplied midpoint values\n      as described above. A `Tensor` of the same dtype as `values` but shape\n      `[n+1]` where `[n]` is the shape of `values`. The `i`th component of the\n      is the value associated to the time point `t_{i+1}` with `t_0 = 0`.\n  "
    with tf.compat.v1.name_scope(name, default_name='interpolate_adjacent', values=[times, values]):
        dt1 = diff(times, order=1, exclusive=False)
        dt2 = diff(times, order=2, exclusive=False)[1:]
        weight_right = dt1[:-1] / dt2
        weight_left = dt1[1:] / dt2
        interior_values = weight_right * values[1:] + weight_left * values[:-1]
        value_0 = values[0] - 0.5 * (interior_values[0] - values[0])
        value_n = values[-1] - 0.5 * (interior_values[-1] - values[-1])
        return tf.concat([[value_0], interior_values, [value_n]], axis=0)

def _region_1(g1plus2g0, g0plus2g1, g0, g1, x):
    if False:
        print('Hello World!')
    'Computes conditional and value for points in region 1.'
    is_region_1 = (g1plus2g0 < 0) & (g0plus2g1 >= 0) | (g1plus2g0 > 0) & (g0plus2g1 <= 0)
    (a, b, c) = (3 * (g0 + g1), -2 * g1plus2g0, g0)
    region_1_value = (a * x + b) * x + c
    one_minus_x = 1 - x
    integrated_value = x * one_minus_x * (g0 * one_minus_x - g1 * x)
    return (is_region_1, region_1_value, integrated_value)

def _region_2(g1plus2g0, g0plus2g1, g0, g1, x):
    if False:
        i = 10
        return i + 15
    'Computes conditional and value for points in region 2.'
    del g0plus2g1
    is_region_2 = (g0 < 0) & (g1plus2g0 >= 0) | (g0 > 0) & (g1plus2g0 <= 0)
    eta = g1plus2g0 / (g1 - g0)
    x_floor = tf.math.maximum(x, eta)
    ratio = (x_floor - eta) / (1 - eta)
    region_2_value = g0 + (g1 - g0) * tf.math.square(ratio)
    coeff = (g1 - g0) * (1 - eta) / 3
    integrated_value = g0 * x + coeff * ratio ** 3
    return (is_region_2, region_2_value, integrated_value)

def _region_3(g1plus2g0, g0plus2g1, g0, g1, x):
    if False:
        while True:
            i = 10
    'Computes conditional and value for points in region 3.'
    del g1plus2g0
    is_region_3 = (g1 <= 0) & (g0plus2g1 > 0) | (g1 >= 0) & (g0plus2g1 < 0)
    eta = 3 * g1 / (g1 - g0)
    x_cap = tf.math.minimum(x, eta)
    ratio = (eta - x_cap) / eta
    ratio = tf.where(tf.math.is_nan(ratio), tf.zeros_like(ratio), ratio)
    region_3_value = g1 + (g0 - g1) * tf.math.square(ratio)
    integrated_value = g1 * x + eta * (g0 - g1) / 3 * (1 - ratio ** 3)
    return (is_region_3, region_3_value, integrated_value)

def _region_4(g1plus2g0, g0plus2g1, g0, g1, x):
    if False:
        while True:
            i = 10
    'Computes conditional and value for points in region 4.'
    del g1plus2g0, g0plus2g1
    is_region_4 = (g0 >= 0) & (g1 > 0) | (g0 <= 0) & (g1 < 0)
    eta = g1 / (g0 + g1)
    x_cap = tf.math.minimum(x, eta)
    x_floor = tf.math.maximum(x, eta)
    shift = -0.5 * (eta * g0 + (1 - eta) * g1)
    ratio_cap = (eta - x_cap) / eta
    ratio_floor = (x_floor - eta) / (1 - eta)
    region_4_value = shift + (g0 - shift) * tf.math.square(ratio_cap) + (g1 - shift) * tf.math.square(ratio_floor)
    integrated_value = shift * x + (g0 - shift) * eta * (1 - ratio_cap ** 3) / 3 + (g1 - shift) * (1 - eta) * ratio_floor ** 3 / 3
    return (is_region_4, region_4_value, integrated_value)
__all__ = ['diff', 'interpolate', 'interpolate_forward_rate', 'interpolate_yields']