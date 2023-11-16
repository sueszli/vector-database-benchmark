"""Collection of functions to compute properties of cashflows."""
import tensorflow.compat.v2 as tf

def present_value(cashflows, discount_factors, dtype=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes present value of a stream of cashflows given discount factors.\n\n\n  ```python\n\n    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.\n    # Note that the first four entries in the cashflows are the cashflows of\n    # the first bond (group=0) and the next six are the cashflows of the second\n    # bond (group=1).\n    cashflows = [[20, 20, 20, 1020, 0, 0],\n                 [30, 30, 30, 30, 30, 1030]]\n\n    # Corresponding discount factors for the cashflows\n    discount_factors = [[0.96, 0.93, 0.9, 0.87, 1.0, 1.0],\n                        [0.97, 0.95, 0.93, 0.9, 0.88, 0.86]]\n\n    present_values = present_value(\n        cashflows, discount_factors, dtype=np.float64)\n    # Expected: [943.2, 1024.7]\n  ```\n\n  Args:\n    cashflows: A real `Tensor` of shape `batch_shape + [n]`. The set of\n      cashflows of underlyings. `n` is the number of cashflows per bond\n      and `batch_shape` is the number of bonds. Bonds with different number\n      of cashflows should be padded to a common number `n`.\n    discount_factors: A `Tensor` of the same `dtype` as `cashflows` and of\n      compatible shape. The set of discount factors corresponding to the\n      cashflows.\n    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.\n      Default value: None which maps to the default dtype inferred from\n      `cashflows`.\n    name: Python str. The name to give to the ops created by this function.\n      Default value: None which maps to 'present_value'.\n\n  Returns:\n    Real `Tensor` of shape `batch_shape`. The present values of the cashflows.\n  "
    name = name or 'present_value'
    with tf.name_scope(name):
        cashflows = tf.convert_to_tensor(cashflows, dtype=dtype, name='cashflows')
        dtype = dtype or cashflows.dtype
        discount_factors = tf.convert_to_tensor(discount_factors, dtype=dtype, name='discount_factors')
        discounted = cashflows * discount_factors
        return tf.math.reduce_sum(discounted, axis=-1)

def pv_from_yields(cashflows, times, yields, groups=None, dtype=None, name=None):
    if False:
        while True:
            i = 10
    "Computes present value of cashflows given yields.\n\n  For a more complete description of the terminology as well as the mathematics\n  of pricing bonds, see Ref [1]. In particular, note that `yields` here refers\n  to the yield of the bond as defined in Section 4.4 of Ref [1]. This is\n  sometimes also referred to as the internal rate of return of a bond.\n\n  #### Example\n\n  The following example demonstrates the present value computation for two\n  bonds. Both bonds have 1000 face value with semi-annual coupons. The first\n  bond has 4% coupon rate and 2 year expiry. The second has 6% coupon rate and\n  3 year expiry. The yields to maturity (ytm) are 7% and 5% respectively.\n\n  ```python\n    dtype = np.float64\n\n    # The first element is the ytm of the first bond and the second is the\n    # yield of the second bond.\n    yields_to_maturity = np.array([0.07, 0.05], dtype=dtype)\n\n    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.\n    # Note that the first four entries in the cashflows are the cashflows of\n    # the first bond (group=0) and the next six are the cashflows of the second\n    # bond (group=1).\n    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],\n                         dtype=dtype)\n\n    # The times of the cashflows.\n    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)\n\n    # Group entries take values between 0 and 1 (inclusive) as there are two\n    # bonds. One needs to assign each of the cashflow entries to one group or\n    # the other.\n    groups = np.array([0] * 4 + [1] * 6)\n\n    # Produces [942.712, 1025.778] as the values of the two bonds.\n    present_values = pv_from_yields(\n        cashflows, times, yields_to_maturity, groups=groups, dtype=dtype)\n  ```\n\n  #### References:\n\n  [1]: John C. Hull. Options, Futures and Other Derivatives. Ninth Edition.\n    June 2006.\n\n  Args:\n    cashflows: Real rank 1 `Tensor` of size `n`. The set of cashflows underlying\n      the bonds.\n    times: Real positive rank 1 `Tensor` of size `n`. The set of times at which\n      the corresponding cashflows occur quoted in years.\n    yields: Real rank 1 `Tensor` of size `1` if `groups` is None or of size `k`\n      if the maximum value in the `groups` is of `k-1`. The continuously\n      compounded yields to maturity/internal rate of returns corresponding to\n      each of the cashflow groups. The `i`th component is the yield to apply to\n      all the cashflows with group label `i` if `groups` is not None. If\n      `groups` is None, then this is a `Tensor` of size `[1]` and the only\n      component is the yield that applies to all the cashflows.\n    groups: Optional int `Tensor` of size `n` containing values between 0 and\n      `k-1` where `k` is the number of related cashflows.\n      Default value: None. This implies that all the cashflows are treated as a\n        single group.\n    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.\n      Default value: None which maps to the default dtype inferred from\n      `cashflows`.\n    name: Python str. The name to give to the ops created by this function.\n      Default value: None which maps to 'pv_from_yields'.\n\n  Returns:\n    Real rank 1 `Tensor` of size `k` if groups is not `None` else of size `[1]`.\n      The present value of the cashflows. The `i`th component is the present\n      value of the cashflows in group `i` or to the entirety of the cashflows\n      if `groups` is None.\n  "
    with tf.compat.v1.name_scope(name, default_name='pv_from_yields', values=[cashflows, times, yields, groups]):
        cashflows = tf.convert_to_tensor(cashflows, dtype=dtype, name='cashflows')
        times = tf.convert_to_tensor(times, dtype=dtype, name='times')
        yields = tf.convert_to_tensor(yields, dtype=dtype, name='yields')
        cashflow_yields = yields
        if groups is not None:
            groups = tf.convert_to_tensor(groups, name='groups')
            cashflow_yields = tf.gather(yields, groups)
        discounted = cashflows * tf.math.exp(-times * cashflow_yields)
        if groups is not None:
            return tf.math.segment_sum(discounted, groups)
        return tf.math.reduce_sum(discounted, keepdims=True)

def yields_from_pv(cashflows, times, present_values, groups=None, tolerance=1e-08, max_iterations=10, dtype=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes yields to maturity from present values of cashflows.\n\n  For a complete description of the terminology as well as the mathematics\n  of computing bond yields, see Ref [1]. Note that `yields` here refers\n  to the yield of the bond as defined in Section 4.4 of Ref [1]. This is\n  sometimes also referred to as the internal rate of return of a bond.\n\n  #### Example\n\n  The following example demonstrates the yield computation for two\n  bonds. Both bonds have 1000 face value with semi-annual coupons. The first\n  bond has 4% coupon rate and 2 year expiry. The second has 6% coupon rate and\n  3 year expiry. The true yields to maturity (ytm) are 7% and 5% respectively.\n\n  ```python\n    dtype = np.float64\n\n    # The first element is the present value (PV) of the first bond and the\n    # second is the PV of the second bond.\n    present_values = np.array([942.71187528177757, 1025.7777300221542],\n                              dtype=dtype)\n\n    # 2 and 3 year bonds with 1000 face value and 4%, 6% semi-annual coupons.\n    # Note that the first four entries in the cashflows are the cashflows of\n    # the first bond (group=0) and the next six are the cashflows of the second\n    # bond (group=1).\n    cashflows = np.array([20, 20, 20, 1020, 30, 30, 30, 30, 30, 1030],\n                         dtype=dtype)\n\n    # The times of the cashflows.\n    times = np.array([0.5, 1, 1.5, 2, 0.5, 1, 1.50, 2, 2.5, 3], dtype=dtype)\n\n    # Group entries take values between 0 and 1 (inclusive) as there are two\n    # bonds. One needs to assign each of the cashflow entries to one group or\n    # the other.\n    groups = np.array([0] * 4 + [1] * 6)\n\n    # Expected yields = [0.07, 0.05]\n    yields = yields_from_pv(\n        cashflows, times, present_values, groups=groups, dtype=dtype)\n  ```\n\n  #### References:\n\n  [1]: John C. Hull. Options, Futures and Other Derivatives. Ninth Edition.\n    June 2006.\n\n  Args:\n    cashflows: Real rank 1 `Tensor` of size `n`. The set of cashflows underlying\n      the bonds.\n    times: Real positive rank 1 `Tensor` of size `n`. The set of times at which\n      the corresponding cashflows occur quoted in years.\n    present_values: Real rank 1 `Tensor` of size `k` where `k-1` is the maximum\n      value in the `groups` arg if supplied. If `groups` is not supplied, then\n      this is a `Tensor` of size `1`. The present values corresponding to each\n      of the cashflow groups. The `i`th component is the present value of all\n      the cashflows with group label `i` (or the present value of all the\n      cashflows if `groups=None`).\n    groups: Optional int `Tensor` of size `n` containing values between 0 and\n      `k-1` where `k` is the number of related cashflows.\n      Default value: None. This implies that all the cashflows are treated as a\n        single group.\n    tolerance: Positive real scalar `Tensor`. The tolerance for the estimated\n      yields. The yields are computed using a Newton root finder. The iterations\n      stop when the inferred yields change by less than this tolerance or the\n      maximum iterations are exhausted (whichever is earlier).\n      Default value: 1e-8.\n    max_iterations: Positive scalar int `Tensor`. The maximum number of\n      iterations to use to compute the yields. The iterations stop when the max\n      iterations is exhausted or the tolerance is reached (whichever is\n      earlier). Supply `None` to remove the limit on the number of iterations.\n      Default value: 10.\n    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.\n      Default value: None which maps to the default dtype inferred from\n      `cashflows`.\n    name: Python str. The name to give to the ops created by this function.\n      Default value: None which maps to 'yields_from_pv'.\n\n  Returns:\n    Real rank 1 `Tensor` of size `k`. The yield to maturity of the cashflows.\n      The `i`th component is the yield to maturity of the cashflows in group\n      `i`.\n  "
    with tf.compat.v1.name_scope(name, default_name='yields_from_pv', values=[cashflows, times, present_values, groups, tolerance, max_iterations]):
        cashflows = tf.convert_to_tensor(cashflows, dtype=dtype, name='cashflows')
        times = tf.convert_to_tensor(times, dtype=dtype, name='times')
        present_values = tf.convert_to_tensor(present_values, dtype=dtype, name='present_values')
        if groups is None:
            groups = tf.zeros_like(cashflows, dtype=tf.int32, name='groups')
        else:
            groups = tf.convert_to_tensor(groups, name='groups')

        def pv_and_duration(yields):
            if False:
                return 10
            cashflow_yields = tf.gather(yields, groups)
            discounted = cashflows * tf.math.exp(-times * cashflow_yields)
            durations = tf.math.segment_sum(discounted * times, groups)
            pvs = tf.math.segment_sum(discounted, groups)
            return (pvs, durations)
        yields0 = tf.zeros_like(present_values)

        def _cond(should_stop, yields):
            if False:
                i = 10
                return i + 15
            del yields
            return tf.math.logical_not(should_stop)

        def _body(should_stop, yields):
            if False:
                i = 10
                return i + 15
            del should_stop
            (pvs, durations) = pv_and_duration(yields)
            delta_yields = (pvs - present_values) / durations
            next_should_stop = tf.math.reduce_max(tf.abs(delta_yields)) <= tolerance
            return (next_should_stop, yields + delta_yields)
        loop_vars = (tf.convert_to_tensor(False), yields0)
        (_, estimated_yields) = tf.while_loop(_cond, _body, loop_vars, shape_invariants=(tf.TensorShape([]), tf.TensorShape([None])), maximum_iterations=max_iterations, parallel_iterations=1)
        return estimated_yields
__all__ = ['present_value', 'pv_from_yields', 'yields_from_pv']