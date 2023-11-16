"""Utility functions needed for brownian motion and related processes."""
import tensorflow.compat.v2 as tf
from tf_quant_finance.math import gradient

def is_callable(var_or_fn):
    if False:
        while True:
            i = 10
    'Returns whether an object is callable or not.'
    if hasattr(var_or_fn, '__call__'):
        return True
    try:
        return callable(var_or_fn)
    except NameError:
        return False

def outer_multiply(x, y):
    if False:
        i = 10
        return i + 15
    'Performs an outer multiplication of two tensors.\n\n  Given two `Tensor`s, `S` and `T` of shape `s` and `t` respectively, the outer\n  product `P` is a `Tensor` of shape `s + t` whose components are given by:\n\n  ```none\n  P_{i1,...ik, j1, ... , jm} = S_{i1...ik} T_{j1, ... jm}\n  ```\n\n  Args:\n    x: A `Tensor` of any shape and numeric dtype.\n    y: A `Tensor` of any shape and the same dtype as `x`.\n\n  Returns:\n    outer_product: A `Tensor` of shape Shape[x] + Shape[y] and the same dtype\n      as `x`.\n  '
    x_shape = tf.shape(x)
    padded_shape = tf.concat([x_shape, tf.ones(tf.rank(y), dtype=x_shape.dtype)], axis=0)
    return tf.reshape(x, padded_shape) * y

def construct_drift_data(drift, total_drift_fn, dim, dtype):
    if False:
        return 10
    'Constructs drift functions.'
    if total_drift_fn is None:
        if drift is None:
            return _default_drift_data(dim, dtype)
        if is_callable(drift):
            return (drift, None)

        def total_drift(t1, t2):
            if False:
                while True:
                    i = 10
            dt = tf.convert_to_tensor(t2 - t1, dtype=dtype)
            return outer_multiply(dt, tf.ones([dim], dtype=dt.dtype) * drift)
        return (_make_drift_fn_from_const(drift, dim, dtype), total_drift)
    if drift is None:

        def drift_from_total_drift(t):
            if False:
                print('Hello World!')
            start_time = tf.zeros_like(t)
            return gradient.fwd_gradient(lambda x: total_drift_fn(start_time, x), t)
        return (drift_from_total_drift, total_drift_fn)
    if is_callable(drift):
        return (drift, total_drift_fn)
    return (_make_drift_fn_from_const(drift, dim, dtype), total_drift_fn)

def construct_vol_data(volatility, total_covariance_fn, dim, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Constructs volatility data.\n\n  This function resolves the supplied arguments in to the following ten cases:\n  (vol -> volatility, total_covar -> total_covariance_fn)\n  1. vol and total_covar are both None -> Return default values.\n  2. total_covar is supplied and vol is None -> compute vol from total covar.\n  3. total_covar is supplied and vol is a callable -> Return supplied values.\n  4. total_covar is supplied and vol is a scalar constant.\n  5. total_covar is supplied and vol is a vector constant.\n  6. total_covar is supplied and vol is a matrix constant.\n  7. total_covar is not supplied and vol is a callable -> Return None for total\n    covariance function.\n  8. total_covar is not supplied and vol is a scalar constant.\n  9. total_covar is not supplied and vol is a vector constant.\n  10. total_covar is not supplied and vol is a matrix.\n\n  For cases 4, 5 and 6 we create an appropriate volatility fn. For cases 8\n  through to 10 we do the same but also create an appropriate covariance\n  function.\n\n  Args:\n    volatility: The volatility specification. None or a callable or a scalar,\n      vector or matrix.\n    total_covariance_fn: The total covariance function. Either None or a\n      callable.\n    dim: int. The dimension of the process.\n    dtype: The default dtype to use.\n\n  Returns:\n    A tuple of two callables:\n      volatility_fn: A function accepting a time argument and returning\n        the volatility at that time.\n      total_covariance_fn: A function accepting two time arguments and\n        returning the total covariance between the two times.\n  '
    if volatility is None and total_covariance_fn is None:
        return _default_vol_data(dim, dtype)
    if total_covariance_fn is not None:
        if volatility is None:
            vol_fn = _volatility_fn_from_total_covar_fn(total_covariance_fn)
            return (vol_fn, total_covariance_fn)
        if is_callable(volatility):
            return (volatility, total_covariance_fn)
        return _construct_vol_data_const_vol(volatility, total_covariance_fn, dim, dtype)
    if is_callable(volatility):
        return (volatility, None)
    return _construct_vol_data_const_vol(volatility, None, dim, dtype)

def _make_drift_fn_from_const(drift_const, dim, dtype):
    if False:
        i = 10
        return i + 15
    drift_const = tf.convert_to_tensor(drift_const, dtype=dtype)
    drift_const = tf.ones([dim], dtype=drift_const.dtype) * drift_const
    return lambda t: outer_multiply(tf.ones_like(t), drift_const)

def _volatility_fn_from_total_covar_fn(total_covariance_fn):
    if False:
        for i in range(10):
            print('nop')
    'Volatility function from total covariance function.'

    def vol_fn(time):
        if False:
            print('Hello World!')
        start_time = tf.zeros_like(time)
        total_covar_fn = lambda t: total_covariance_fn(start_time, t)
        vol_sq = gradient.fwd_gradient(total_covar_fn, time)
        return tf.linalg.cholesky(vol_sq, name='volatility')
    return vol_fn

def _default_drift_data(dimension, dtype):
    if False:
        return 10
    'Constructs a function which returns a zero drift.'

    def zero_drift(time):
        if False:
            while True:
                i = 10
        shape = tf.concat([tf.shape(time), [dimension]], axis=0)
        time = tf.convert_to_tensor(time, dtype=dtype)
        return tf.zeros(shape, dtype=time.dtype)
    return (zero_drift, lambda t1, t2: zero_drift(t1))

def _default_vol_data(dimension, dtype):
    if False:
        while True:
            i = 10
    'Unit volatility and corresponding covariance functions.'

    def unit_vol(time):
        if False:
            for i in range(10):
                print('nop')
        shape = tf.concat([tf.shape(time), [dimension, dimension]], axis=0)
        out_dtype = tf.convert_to_tensor(time, dtype=dtype).dtype
        return tf.ones(shape, dtype=out_dtype)

    def covar(start_time, end_time):
        if False:
            i = 10
            return i + 15
        dt = tf.convert_to_tensor(end_time - start_time, dtype=dtype, name='dt')
        return outer_multiply(dt, tf.eye(dimension, dtype=dt.dtype))
    return (unit_vol, covar)

def _ensure_matrix(volatility, dim, dtype):
    if False:
        while True:
            i = 10
    'Converts a volatility tensor to the right shape.'
    rank = len(volatility.shape)
    if not rank:
        return tf.eye(dim, dtype=dtype) * volatility
    if rank == 1:
        return tf.linalg.tensor_diag(volatility)
    return volatility

def _covar_from_vol(volatility, dim, dtype):
    if False:
        while True:
            i = 10
    rank = len(volatility.shape)
    if not rank:
        return volatility * volatility * tf.eye(dim, dtype=dtype)
    if rank == 1:
        return tf.linalg.tensor_diag(volatility * volatility)
    return tf.linalg.matmul(volatility, volatility, transpose_b=True)

def _construct_vol_data_const_vol(volatility, total_covariance_fn, dim, dtype):
    if False:
        i = 10
        return i + 15
    'Constructs vol data when constant volatility is supplied.'
    volatility_matrix = _ensure_matrix(volatility, dim, dtype)

    def vol_fn(time):
        if False:
            while True:
                i = 10
        return outer_multiply(tf.ones_like(time), volatility_matrix)
    if total_covariance_fn is not None:
        return (vol_fn, total_covariance_fn)
    covariance_matrix = _covar_from_vol(volatility, dim, dtype)
    covar_fn = lambda t1, t2: outer_multiply(t2 - t1, covariance_matrix)
    return (vol_fn, covar_fn)