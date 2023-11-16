"""Methods to construct a swap curve.

Building swap curves is a core problem in mathematical finance. Swap
curves are built using the available market data in liquidly traded fixed income
products. These include LIBOR rates, interest rate swaps, forward rate
agreements (FRAs) or exchange traded futures contracts. This module contains
methods to build swap curve from market data.

The algorithm implemented here uses conjugate gradient optimization to minimize
the weighted least squares error between the input present values of the
instruments and the present values computed using the constructed swap curve.

#### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 6. 2010.
"""
from typing import List, Callable, Any
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.math import make_val_and_grad_fn
from tf_quant_finance.math import optimizer as optimizers
from tf_quant_finance.math.interpolation import linear
from tf_quant_finance.rates import swap_curve_common as scc
__all__ = ['swap_curve_fit']

def swap_curve_fit(float_leg_start_times: List[types.RealTensor], float_leg_end_times: List[types.RealTensor], float_leg_daycount_fractions: List[types.RealTensor], fixed_leg_start_times: List[types.RealTensor], fixed_leg_end_times: List[types.RealTensor], fixed_leg_daycount_fractions: List[types.RealTensor], fixed_leg_cashflows: List[types.RealTensor], present_values: List[types.RealTensor], initial_curve_rates: types.RealTensor, present_values_settlement_times: List[types.RealTensor]=None, float_leg_discount_rates: List[types.RealTensor]=None, float_leg_discount_times: List[types.RealTensor]=None, fixed_leg_discount_rates: List[types.RealTensor]=None, fixed_leg_discount_times: List[types.RealTensor]=None, optimizer: Callable[..., Any]=None, curve_interpolator: Callable[..., types.RealTensor]=None, instrument_weights: types.RealTensor=None, curve_tolerance: types.RealTensor=1e-08, maximum_iterations: types.IntTensor=50, dtype: tf.DType=None, name: str=None) -> scc.SwapCurveBuilderResult:
    if False:
        i = 10
        return i + 15
    "Constructs the zero swap curve using optimization.\n\n  A zero swap curve is a function of time which gives the interest rate that\n  can be used to project forward rates at arbitrary `t` for the valuation of\n  interest rate securities.\n\n  Suppose we have a set of `N` Interest Rate Swaps (IRS) `S_i` with increasing\n  expiries whose market prices are known.\n  Suppose also that the `i`th IRS issues cashflows at times `T_{ij}` where\n  `1 <= j <= n_i` and `n_i` is the number of cashflows (including expiry)\n  for the `i`th swap.\n  Denote by `T_i` the time of final payment for the `i`th swap\n  (hence `T_i = T_{i,n_i}`). This function estimates a set of rates `r(T_i)`\n  such that when these rates are interpolated to all other cashflow times,\n  the computed value of the swaps matches the market value of the swaps\n  (within some tolerance). Rates at intermediate times are interpolated using\n  the user specified interpolation method (the default interpolation method\n  is linear interpolation on rates).\n\n  #### Example:\n\n  The following example illustrates the usage by building an implied swap curve\n  from four vanilla (fixed to float) LIBOR swaps.\n\n  ```python\n\n  dtype = np.float64\n\n  # Next we will set up LIBOR reset and payment times for four spot starting\n  # swaps with maturities 1Y, 2Y, 3Y, 4Y. The LIBOR rate spans 6M.\n\n  float_leg_start_times = [\n            np.array([0., 0.5], dtype=dtype),\n            np.array([0., 0.5, 1., 1.5], dtype=dtype),\n            np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5], dtype=dtype),\n            np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=dtype)\n        ]\n\n  float_leg_end_times = [\n            np.array([0.5, 1.0], dtype=dtype),\n            np.array([0.5, 1., 1.5, 2.0], dtype=dtype),\n            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),\n            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)\n        ]\n\n  # Next we will set up start and end times for semi-annual fixed coupons.\n\n  fixed_leg_start_times = [\n            np.array([0., 0.5], dtype=dtype),\n            np.array([0., 0.5, 1., 1.5], dtype=dtype),\n            np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5], dtype=dtype),\n            np.array([0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=dtype)\n        ]\n\n  fixed_leg_end_times = [\n            np.array([0.5, 1.0], dtype=dtype),\n            np.array([0.5, 1., 1.5, 2.0], dtype=dtype),\n            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=dtype),\n            np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)\n        ]\n\n  # Next setup a trivial daycount for floating and fixed legs.\n\n  float_leg_daycount = [\n            np.array([0.5, 0.5], dtype=dtype),\n            np.array([0.5, 0.5, 0.5, 0.5], dtype=dtype),\n            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype),\n            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype)\n        ]\n\n  fixed_leg_daycount = [\n            np.array([0.5, 0.5], dtype=dtype),\n            np.array([0.5, 0.5, 0.5, 0.5], dtype=dtype),\n            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype),\n            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype)\n        ]\n\n  fixed_leg_cashflows = [\n        # 1 year swap with 2.855% semi-annual fixed payments.\n        np.array([-0.02855, -0.02855], dtype=dtype),\n        # 2 year swap with 3.097% semi-annual fixed payments.\n        np.array([-0.03097, -0.03097, -0.03097, -0.03097], dtype=dtype),\n        # 3 year swap with 3.1% semi-annual fixed payments.\n        np.array([-0.031, -0.031, -0.031, -0.031, -0.031, -0.031], dtype=dtype),\n        # 4 year swap with 3.2% semi-annual fixed payments.\n        np.array([-0.032, -0.032, -0.032, -0.032, -0.032, -0.032, -0.032,\n        -0.032], dtype=dtype)\n    ]\n\n  # The present values of the above IRS.\n  pvs = np.array([0., 0., 0., 0.], dtype=dtype)\n\n  # Initial state of the curve.\n  initial_curve_rates = np.array([0.01, 0.01, 0.01, 0.01], dtype=dtype)\n\n  results = swap_curve_fit(float_leg_start_times, float_leg_end_times,\n                           float_leg_daycount, fixed_leg_start_times,\n                           fixed_leg_end_times, fixed_leg_cashflows,\n                           fixed_leg_daycount, pvs, dtype=dtype,\n                           initial_curve_rates=initial_curve_rates)\n\n  #### References:\n  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,\n      Volume I: Foundations and Vanilla Models. Chapter 6. 2010.\n\n  Args:\n    float_leg_start_times: List of `Tensor`s. Each `Tensor` must be either of\n      shape `batch_shape  + [k_i]` or `[k_i]` and of the same real dtype. `k_i`\n      may be of different sizes. Each `Tensor` represents the beginning of the\n      accrual period for the forward rate which determines the floating payment.\n      Each element in the list belong to a unique swap to be used to build the\n      curve.\n    float_leg_end_times: List of `Tensor`s of shapes and `dtype` compatible with\n      `float_leg_start_times`. Each `Tensor` represents the end of the\n      accrual period for the forward rate which determines the floating payment.\n    float_leg_daycount_fractions: List of `Tensor`s of shapes and `dtype`\n      compatible with `float_leg_start_times`. Each `Tensor` represents the\n      daycount fraction of the forward rate which determines the floating\n      payment.\n    fixed_leg_start_times: List of `Tensor`s. Each `Tensor` must be either of\n      shape `batch_shape  + [n_i]` or `[n_i]` and of the same real dtype.\n      `n_i` may be of different sizes. All elements must have the same `dtype`\n      as `float_leg_start_times`. Each `Tensor` represents the beginning of the\n      accrual period fixed coupon.\n    fixed_leg_end_times: List of `Tensor`s of shapes and `dtype` compatible with\n      `fixed_leg_start_times`. All elements must have the same `dtype` as\n      `fixed_leg_start_times`. Each `Tensor` represents the\n      end of the accrual period for the fixed coupon.\n    fixed_leg_daycount_fractions: List of `Tensor`s of shapes and `dtype`\n    compatible with\n      `fixed_leg_start_times` Each `Tensor` represents the daycount fraction\n      applicable for the fixed payment.\n    fixed_leg_cashflows: List of `Tensor`s of shapes and `dtype` compatible with\n      `fixed_leg_start_times`. The input contains fixed cashflows at each\n      coupon payment time including notional (if any). The sign should be\n      negative (positive) to indicate net outgoing (incoming) cashflow.\n    present_values: List containing `Tensor`s of the same dtype as\n      elements of `fixed_leg_cashflows` and of shapes compatible with\n      `batch_shape`. The length of the list must be the same as the length of\n      `fixed_leg_cashflows`. The input contains the market price of the\n      underlying instruments.\n    initial_curve_rates: A `Tensor` of the `dtype` as `present_values` and of\n      shape `[batch_shape, num_instruments]` where `num_instruments` is the\n      length of `float_leg_start_times`. The starting guess for the discount\n      rates used to initialize the iterative procedure.\n    present_values_settlement_times: Optional list of `Tensor`s with the shapes\n      and `dtype` compatible with `present_values` The settlement times for the\n      present values is the time from now when the instrument is traded to the\n      time that the purchase price is actually delivered. If not supplied, then\n      it is assumed that the settlement times are zero for every bond.\n      Default value: `None` which is equivalent to zero settlement times.\n    float_leg_discount_rates: Optional list of `Tensor`s with the shapes\n      and `dtype` compatible with `present_values`. This input contains the\n      continuously compounded discount rates the will be used\n      to discount the floating cashflows. This allows the swap curve to\n      constructed using an independent discount curve (e.g. OIS curve). By\n      default the cashflows are discounted using the curve that is being\n      constructed.\n    float_leg_discount_times: Optional list of `Tensor`s with the shapes\n      and `dtype` compatible with `present_values`. This input contains the\n      times corresponding to the rates specified via\n      the `float_leg_discount_rates`.\n    fixed_leg_discount_rates: Optional list of `Tensor`s with the shapes\n      and `dtype` compatible with `present_values`. This input contains the\n      continuously compounded discount rates the will be used to discount the\n      fixed cashflows. This allows the swap curve to constructed using an\n      independent discount curve (e.g. OIS curve). By default the cashflows are\n      discounted using the curve that is being constructed.\n    fixed_leg_discount_times: Optional list of `Tensor`s with the shapes\n      and `dtype` compatible with `present_values`. This input contains the\n      times corresponding to the rates specified via the\n      `fixed_leg_discount_rates`.\n    optimizer: Optional Python callable which implements the algorithm used\n      to minimize the objective function during calibration. It should have\n      the following interface: result =\n        optimizer(value_and_gradients_function, initial_position, tolerance,\n        max_iterations) `value_and_gradients_function` is a Python callable that\n        accepts a point as a real `Tensor` and returns a tuple of `Tensor`s of\n        real dtype containing the value of the function and its gradient at that\n        point. 'initial_position' is a real `Tensor` containing the starting\n        point of the optimization, 'tolerance' is a real scalar `Tensor` for\n        stopping tolerance for the procedure and `max_iterations` specifies the\n        maximum number of iterations.\n      `optimizer` should return a namedtuple containing the items: `position`\n        (a tensor containing the optimal value), `converged` (a boolean\n        indicating whether the optimize converged according the specified\n        criteria), `failed` (a boolean indicating if the optimization resulted\n        in a failure), `num_iterations` (the number of iterations used), and\n        `objective_value` ( the value of the objective function at the optimal\n        value). The default value for `optimizer` is None and conjugate\n        gradient algorithm is used.\n      Default value: `None` - indicating conjugate gradient minimizer.\n    curve_interpolator: Optional Python callable used to interpolate the zero\n      swap rates at cashflow times. It should have the following interface:\n      yi = curve_interpolator(xi, x, y)\n      `x`, `y`, 'xi', 'yi' are all `Tensors` of real dtype. `x` and `y` are the\n      sample points and values (respectively) of the function to be\n      interpolated. `xi` are the points at which the interpolation is\n      desired and `yi` are the corresponding interpolated values returned by the\n      function. The default value for `curve_interpolator` is None in which\n      case linear interpolation is used.\n      Default value: `None`. If not supplied, the yields to maturity for the\n        bonds is used as the initial value.\n    instrument_weights: Optional 'Tensor' of the same dtype and shape as\n      `initial_curve_rates`. This input contains the weight of each instrument\n      in computing the objective function for the conjugate gradient\n      optimization. By default the weights are set to be the inverse of\n      maturities.\n    curve_tolerance: Optional positive scalar `Tensor` of same dtype as\n      elements of `bond_cashflows`. The absolute tolerance for terminating the\n      iterations used to fit the rate curve. The iterations are stopped when the\n      estimated discounts at the expiry times of the bond_cashflows change by a\n      amount smaller than `discount_tolerance` in an iteration.\n      Default value: 1e-8.\n    maximum_iterations: Optional positive integer `Tensor`. The maximum number\n      of iterations permitted when fitting the curve.\n      Default value: 50.\n    dtype: `tf.Dtype`. If supplied the dtype for the (elements of)\n      `float_leg_start_times`, and `fixed_leg_start_times`.\n      Default value: None which maps to the default dtype inferred by\n      TensorFlow.\n    name: Python str. The name to give to the ops created by this function.\n      Default value: `None` which maps to 'swap_curve'.\n\n  Returns:\n    curve_builder_result: An instance of `SwapCurveBuilderResult` containing the\n      following attributes.\n      times: Rank 1 real `Tensor`. Times for the computed discount rates. These\n        are chosen to be the expiry times of the supplied cashflows.\n      discount_rates: Rank 1 `Tensor` of the same dtype as `times`.\n        The inferred discount rates.\n      discount_factor: Rank 1 `Tensor` of the same dtype as `times`.\n        The inferred discount factors.\n      initial_discount_rates: Rank 1 `Tensor` of the same dtype as `times`. The\n        initial guess for the discount rates.\n      converged: Scalar boolean `Tensor`. Whether the procedure converged.\n        The procedure is said to have converged when the maximum absolute\n        difference in the discount factors from one iteration to the next falls\n        below the `discount_tolerance`.\n      failed: Scalar boolean `Tensor`. Whether the procedure failed. Procedure\n        may fail either because a NaN value was encountered for the discount\n        rates or the discount factors.\n      iterations: Scalar int32 `Tensor`. Number of iterations performed.\n      objective_value: Scalar real `Tensor`. The value of the ibjective function\n        evaluated using the fitted swap curve.\n\n  Raises:\n    ValueError: If the initial state of the curve is not\n    supplied to the function.\n\n  "
    with tf.name_scope(name or 'swap_curve'):
        if optimizer is None:
            optimizer = optimizers.conjugate_gradient_minimize
        present_values = _convert_to_tensors(dtype, present_values, 'present_values')
        dtype = present_values[0].dtype
        if present_values_settlement_times is None:
            pv_settlement_times = [tf.zeros([], dtype=dtype) for pv in present_values]
        else:
            pv_settlement_times = _convert_to_tensors(dtype, present_values_settlement_times, 'pv_settlement_times')
        float_leg_start_times = _convert_to_tensors(dtype, float_leg_start_times, 'float_leg_start_times')
        float_leg_end_times = _convert_to_tensors(dtype, float_leg_end_times, 'float_leg_end_times')
        float_leg_daycount_fractions = _convert_to_tensors(dtype, float_leg_daycount_fractions, 'float_leg_daycount_fractions')
        fixed_leg_start_times = _convert_to_tensors(dtype, fixed_leg_start_times, 'fixed_leg_start_times')
        fixed_leg_end_times = _convert_to_tensors(dtype, fixed_leg_end_times, 'fixed_leg_end_times')
        fixed_leg_daycount_fractions = _convert_to_tensors(dtype, fixed_leg_daycount_fractions, 'fixed_leg_daycount_fractions')
        fixed_leg_cashflows = _convert_to_tensors(dtype, fixed_leg_cashflows, 'fixed_leg_cashflows')
        present_values = tf.stack(present_values, axis=-1)
        if instrument_weights is None:
            instrument_weights = _initialize_instrument_weights(float_leg_end_times, fixed_leg_end_times, dtype=dtype)
        else:
            instrument_weights = _convert_to_tensors(dtype, instrument_weights, 'instrument_weights')
        if curve_interpolator is None:

            def default_interpolator(xi, x, y):
                if False:
                    while True:
                        i = 10
                return linear.interpolate(xi, x, y, dtype=dtype)
            curve_interpolator = default_interpolator
        self_discounting_float_leg = False
        self_discounting_fixed_leg = False
        if float_leg_discount_rates is None and fixed_leg_discount_rates is None:
            self_discounting_float_leg = True
            self_discounting_fixed_leg = True
            float_leg_discount_rates = [0.0]
            float_leg_discount_times = [0.0]
            fixed_leg_discount_rates = [0.0]
            fixed_leg_discount_times = [0.0]
        elif fixed_leg_discount_rates is None:
            fixed_leg_discount_rates = float_leg_discount_rates
            fixed_leg_discount_times = float_leg_discount_times
        elif float_leg_discount_rates is None:
            self_discounting_float_leg = True
            float_leg_discount_rates = [0.0]
            float_leg_discount_times = [0.0]
        float_leg_discount_rates = _convert_to_tensors(dtype, float_leg_discount_rates, 'float_disc_rates')
        float_leg_discount_rates = tf.stack(float_leg_discount_rates, axis=-1)
        float_leg_discount_times = _convert_to_tensors(dtype, float_leg_discount_times, 'float_disc_times')
        float_leg_discount_times = tf.stack(float_leg_discount_times, axis=-1)
        fixed_leg_discount_rates = _convert_to_tensors(dtype, fixed_leg_discount_rates, 'fixed_disc_rates')
        fixed_leg_discount_rates = tf.stack(fixed_leg_discount_rates, axis=-1)
        fixed_leg_discount_times = _convert_to_tensors(dtype, fixed_leg_discount_times, 'fixed_disc_times')
        fixed_leg_discount_times = tf.stack(fixed_leg_discount_times, axis=-1)
        if initial_curve_rates is not None:
            initial_rates = tf.convert_to_tensor(initial_curve_rates, dtype=dtype, name='initial_rates')
        else:
            raise ValueError('Initial state of the curve is not specified.')
        return _build_swap_curve(float_leg_start_times, float_leg_end_times, float_leg_daycount_fractions, fixed_leg_start_times, fixed_leg_end_times, fixed_leg_cashflows, fixed_leg_daycount_fractions, float_leg_discount_rates, float_leg_discount_times, fixed_leg_discount_rates, fixed_leg_discount_times, self_discounting_float_leg, self_discounting_fixed_leg, present_values, pv_settlement_times, optimizer, curve_interpolator, initial_rates, instrument_weights, curve_tolerance, maximum_iterations)

def _build_swap_curve(float_leg_start_times, float_leg_end_times, float_leg_daycount_fractions, fixed_leg_start_times, fixed_leg_end_times, fixed_leg_cashflows, fixed_leg_daycount_fractions, float_leg_discount_rates, float_leg_discount_times, fixed_leg_discount_rates, fixed_leg_discount_times, self_discounting_float_leg, self_discounting_fixed_leg, present_values, pv_settlement_times, optimizer, curve_interpolator, initial_rates, instrument_weights, curve_tolerance, maximum_iterations):
    if False:
        return 10
    'Build the zero swap curve.'
    del fixed_leg_start_times, float_leg_daycount_fractions
    curve_tensors = _create_curve_building_tensors(float_leg_start_times, float_leg_end_times, fixed_leg_end_times, pv_settlement_times)
    expiry_times = curve_tensors.expiry_times
    calc_groups_float = curve_tensors.calc_groups_float
    calc_groups_fixed = curve_tensors.calc_groups_fixed
    settle_times_float = curve_tensors.settle_times_float
    settle_times_fixed = curve_tensors.settle_times_fixed
    float_leg_calc_times_start = tf.concat(float_leg_start_times, axis=-1)
    float_leg_calc_times_end = tf.concat(float_leg_end_times, axis=-1)
    calc_fixed_leg_cashflows = tf.concat(fixed_leg_cashflows, axis=-1)
    calc_fixed_leg_daycount = tf.concat(fixed_leg_daycount_fractions, axis=-1)
    fixed_leg_calc_times = tf.concat(fixed_leg_end_times, axis=-1)

    def _interpolate(x1, x_data, y_data):
        if False:
            print('Hello World!')
        return curve_interpolator(x1, x_data, y_data)

    @make_val_and_grad_fn
    def loss_function(x):
        if False:
            while True:
                i = 10
        'Loss function for the optimization.'
        rates_start = _interpolate(float_leg_calc_times_start, expiry_times, x)
        rates_end = _interpolate(float_leg_calc_times_end, expiry_times, x)
        float_cashflows = tf.math.exp(float_leg_calc_times_end * rates_end - float_leg_calc_times_start * rates_start) - 1.0
        if self_discounting_float_leg:
            float_discount_rates = rates_end
            float_settle_rates = _interpolate(settle_times_float, expiry_times, x)
        else:
            float_discount_rates = _interpolate(float_leg_calc_times_end, float_leg_discount_times, float_leg_discount_rates)
            float_settle_rates = _interpolate(settle_times_float, float_leg_discount_times, float_leg_discount_rates)
        if self_discounting_fixed_leg:
            fixed_discount_rates = _interpolate(fixed_leg_calc_times, expiry_times, x)
            fixed_settle_rates = _interpolate(settle_times_fixed, expiry_times, x)
        else:
            fixed_discount_rates = _interpolate(fixed_leg_calc_times, fixed_leg_discount_times, fixed_leg_discount_rates)
            fixed_settle_rates = _interpolate(settle_times_fixed, fixed_leg_discount_times, fixed_leg_discount_rates)
        calc_discounts_float_leg = tf.math.exp(-float_discount_rates * float_leg_calc_times_end + float_settle_rates * settle_times_float)
        calc_discounts_fixed_leg = tf.math.exp(-fixed_discount_rates * fixed_leg_calc_times + fixed_settle_rates * settle_times_fixed)
        float_pv = tf.linalg.matvec(calc_groups_float, float_cashflows * calc_discounts_float_leg)
        fixed_pv = tf.linalg.matvec(calc_groups_fixed, calc_fixed_leg_daycount * calc_fixed_leg_cashflows * calc_discounts_fixed_leg)
        swap_pv = float_pv + fixed_pv
        value = tf.math.reduce_sum(input_tensor=instrument_weights * (swap_pv - present_values) ** 2, axis=-1)
        return value
    optimization_result = optimizer(loss_function, initial_position=initial_rates, tolerance=curve_tolerance, max_iterations=maximum_iterations)
    discount_rates = optimization_result.position
    discount_factors = tf.math.exp(-discount_rates * expiry_times)
    results = scc.SwapCurveBuilderResult(times=expiry_times, rates=discount_rates, discount_factors=discount_factors, initial_rates=initial_rates, converged=optimization_result.converged, failed=optimization_result.failed, iterations=optimization_result.num_iterations, objective_value=optimization_result.objective_value)
    return results

def _convert_to_tensors(dtype, input_array, name):
    if False:
        i = 10
        return i + 15
    'Converts the supplied list to a tensor.'
    output_tensor = [tf.convert_to_tensor(x, dtype=dtype, name=name + '_{}'.format(i)) for (i, x) in enumerate(input_array)]
    return output_tensor

def _initialize_instrument_weights(float_times, fixed_times, dtype):
    if False:
        return 10
    'Function to compute default initial weights for optimization.'
    weights = tf.ones(len(float_times), dtype=dtype)
    one = tf.ones([], dtype=dtype)
    float_times_last = tf.stack([times[-1] for times in float_times])
    fixed_times_last = tf.stack([times[-1] for times in fixed_times])
    weights = tf.maximum(tf.math.divide_no_nan(one, float_times_last), tf.math.divide_no_nan(one, fixed_times_last))
    weights = tf.minimum(one, weights)
    return tf.unstack(weights, name='instrument_weights')

@utils.dataclass
class CurveFittingVars:
    """Curve fitting variables."""
    expiry_times: types.RealTensor
    calc_groups_float: types.IntTensor
    calc_groups_fixed: types.IntTensor
    settle_times_float: types.RealTensor
    settle_times_fixed: types.RealTensor

def _create_curve_building_tensors(float_leg_start_times, float_leg_end_times, fixed_leg_end_times, pv_settlement_times):
    if False:
        i = 10
        return i + 15
    'Helper function to create tensors needed for curve construction.'
    calc_groups_float = []
    calc_groups_fixed = []
    expiry_times = []
    settle_times_float = []
    settle_times_fixed = []
    num_instruments = len(float_leg_start_times)
    for i in range(num_instruments):
        expiry_times.append(tf.math.maximum(float_leg_end_times[i][-1], fixed_leg_end_times[i][-1]))
        calc_groups_float.append(tf.fill(tf.shape(float_leg_start_times[i]), i))
        calc_groups_fixed.append(tf.fill(tf.shape(fixed_leg_end_times[i]), i))
        settle_time = pv_settlement_times[i]
        if settle_time.shape.rank > 0:
            settle_time = tf.expand_dims(settle_time, axis=-1)
        stf = settle_time + tf.zeros_like(float_leg_start_times[i])
        settle_times_float.append(stf)
        stf = settle_time + tf.zeros_like(fixed_leg_end_times[i])
        settle_times_fixed.append(stf)
    expiry_times = tf.stack(expiry_times, axis=0)
    dtype = expiry_times.dtype
    num_groups_float = len(calc_groups_float)
    calc_groups_float = tf.concat(calc_groups_float, axis=-1)
    axis = calc_groups_float.shape.rank - 1
    calc_groups_float_mat = tf.one_hot(calc_groups_float, num_groups_float, axis=axis, dtype=dtype)
    num_groups_fixed = len(calc_groups_fixed)
    calc_groups_fixed = tf.concat(calc_groups_fixed, axis=-1)
    axis = calc_groups_fixed.shape.rank - 1
    calc_groups_fixed_mat = tf.one_hot(calc_groups_fixed, num_groups_fixed, axis=axis, dtype=dtype)
    settle_times_float = tf.concat(settle_times_float, axis=-1)
    settle_times_fixed = tf.concat(settle_times_fixed, axis=-1)
    return CurveFittingVars(expiry_times=expiry_times, calc_groups_float=calc_groups_float_mat, calc_groups_fixed=calc_groups_fixed_mat, settle_times_float=settle_times_float, settle_times_fixed=settle_times_fixed)