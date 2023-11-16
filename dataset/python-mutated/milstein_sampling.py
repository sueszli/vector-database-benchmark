"""The Milstein sampling method for ito processes."""
import functools
import math
from typing import Callable, List, Optional
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.math import custom_loops
from tf_quant_finance.math import gradient
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.models import utils
_PI = math.pi
_SQRT_2 = np.sqrt(2.0)

def sample(*, dim: int, drift_fn: Callable[..., types.RealTensor], volatility_fn: Callable[..., types.RealTensor], times: types.RealTensor, time_step: Optional[types.RealTensor]=None, num_time_steps: Optional[types.IntTensor]=None, num_samples: types.IntTensor=1, initial_state: Optional[types.RealTensor]=None, grad_volatility_fn: Optional[Callable[..., List[types.RealTensor]]]=None, random_type: Optional[random.RandomType]=None, seed: Optional[types.IntTensor]=None, swap_memory: bool=True, skip: types.IntTensor=0, precompute_normal_draws: bool=True, watch_params: Optional[List[types.RealTensor]]=None, stratonovich_order: int=5, dtype: Optional[tf.DType]=None, name: Optional[str]=None) -> types.RealTensor:
    if False:
        i = 10
        return i + 15
    "Returns a sample paths from the process using the Milstein method.\n\n  For an Ito process,\n\n  ```\n    dX = a(t, X_t) dt + b(t, X_t) dW_t\n  ```\n  given drift `a`, volatility `b` and derivative of volatility `b'`, the\n  Milstein method generates a\n  sequence {Y_n} approximating X\n\n  ```\n  Y_{n+1} = Y_n + a(t_n, Y_n) dt + b(t_n, Y_n) dW_n + \\frac{1}{2} b(t_n, Y_n)\n  b'(t_n, Y_n) ((dW_n)^2 - dt)\n  ```\n  where `dt = t_{n+1} - t_n`, `dW_n = (N(0, t_{n+1}) - N(0, t_n))` and `N` is a\n  sample from the Normal distribution.\n\n  In higher dimensions, when `a(t, X_t)` is a d-dimensional vector valued\n  function and `W_t` is a d-dimensional Wiener process, we have for the kth\n  element of the expansion:\n\n  ```\n  Y_{n+1}[k] = Y_n[k] + a(t_n, Y_n)[k] dt + \\sum_{j=1}^d b(t_n, Y_n)[k, j]\n  dW_n[j] + \\sum_{j_1=1}^d \\sum_{j_2=1}^d L_{j_1} b(t_n, Y_n)[k, j_2] I(j_1,\n  j_2)\n  ```\n  where `L_{j} = \\sum_{i=1}^d b(t_n, Y_n)[i, j] \\frac{\\partial}{\\partial x^i}`\n  is an operator and `I(j_1, j_2) = \\int_{t_n}^{t_{n+1}} \\int_{t_n}^{s_1}\n  dW_{s_2}[j_1] dW_{s_1}[j_2]` is a multiple Ito integral.\n\n\n  See [1] and [2] for details.\n\n  #### References\n  [1]: Wikipedia. Milstein method:\n  https://en.wikipedia.org/wiki/Milstein_method\n  [2]: Peter E. Kloeden,  Eckhard Platen. Numerical Solution of Stochastic\n    Differential Equations. Springer. 1992\n\n  Args:\n    dim: Python int greater than or equal to 1. The dimension of the Ito\n      Process.\n    drift_fn: A Python callable to compute the drift of the process. The\n      callable should accept two real `Tensor` arguments of the same dtype. The\n      first argument is the scalar time t, the second argument is the value of\n      Ito process X - tensor of shape `batch_shape + [dim]`. The result is\n      value of drift a(t, X). The return value of the callable is a real\n      `Tensor` of the same dtype as the input arguments and of shape\n      `batch_shape + [dim]`.\n    volatility_fn: A Python callable to compute the volatility of the process.\n      The callable should accept two real `Tensor` arguments of the same dtype\n      as `times`. The first argument is the scalar time t, the second argument\n      is the value of Ito process X - tensor of shape `batch_shape + [dim]`. The\n      result is value of volatility b(t, X). The return value of the callable is\n      a real `Tensor` of the same dtype as the input arguments and of shape\n      `batch_shape + [dim, dim]`.\n    times: Rank 1 `Tensor` of increasing positive real values. The times at\n      which the path points are to be evaluated.\n    time_step: An optional scalar real `Tensor` - maximal distance between\n      points in grid in Milstein schema.\n      Either this or `num_time_steps` should be supplied.\n      Default value: `None`.\n    num_time_steps: An optional Scalar integer `Tensor` - a total number of time\n      steps performed by the algorithm. The maximal distance between points in\n      grid is bounded by `times[-1] / (num_time_steps - times.shape[0])`.\n      Either this or `time_step` should be supplied.\n      Default value: `None`.\n    num_samples: Positive scalar `int`. The number of paths to draw.\n      Default value: 1.\n    initial_state: `Tensor` of shape `[dim]`. The initial state of the\n      process.\n      Default value: None which maps to a zero initial state.\n    grad_volatility_fn: An optional python callable to compute the gradient of\n      `volatility_fn`. The callable should accept three real `Tensor` arguments\n      of the same dtype as `times`. The first argument is the scalar time t. The\n      second argument is the value of Ito process X - tensor of shape\n      `batch_shape + [dim]`. The third argument is a tensor of input gradients\n      of shape `batch_shape + [dim]` to pass to `gradient.fwd_gradient`. The\n      result is a list of values corresponding to the forward gradient of\n      volatility b(t, X) with respect to X. The return value of the callable is\n      a list of size `dim` containing real `Tensor`s of the same dtype as the\n      input arguments and of shape `batch_shape + [dim, dim]`. Each index of the\n      list corresponds to a dimension of the state. If `None`, the gradient is\n      computed from `volatility_fn` using forward differentiation.\n    random_type: Enum value of `RandomType`. The type of (quasi)-random number\n      generator to use to generate the paths.\n      Default value: None which maps to the standard pseudo-random numbers.\n    seed: Seed for the random number generator. The seed is only relevant if\n      `random_type` is one of `[STATELESS, PSEUDO, HALTON_RANDOMIZED,\n      PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC]`. For `PSEUDO`,\n      `PSEUDO_ANTITHETIC` and `HALTON_RANDOMIZED` the seed should be a Python\n      integer. For `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as\n      an integer `Tensor` of shape `[2]`.\n      Default value: `None` which means no seed is set.\n    swap_memory: A Python bool. Whether GPU-CPU memory swap is enabled for this\n      op. See an equivalent flag in `tf.while_loop` documentation for more\n      details. Useful when computing a gradient of the op since `tf.while_loop`\n      is used to propagate stochastic process in time.\n      Default value: True.\n    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n      Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.\n      Default value: `0`.\n    precompute_normal_draws: Python bool. Indicates whether the noise increments\n      `N(0, t_{n+1}) - N(0, t_n)` are precomputed. For `HALTON` and `SOBOL`\n      random types the increments are always precomputed. While the resulting\n      graph consumes more memory, the performance gains might be significant.\n      Default value: `True`.\n    watch_params: An optional list of zero-dimensional `Tensor`s of the same\n      `dtype` as `initial_state`. If provided, specifies `Tensor`s with respect\n      to which the differentiation of the sampling function will happen. A more\n      efficient algorithm is used when `watch_params` are specified. Note the\n      the function becomes differentiable only wrt to these `Tensor`s and the\n      `initial_state`. The gradient wrt any other `Tensor` is set to be zero.\n    stratonovich_order: A positive integer. The number of terms to use when\n      calculating the approximate Stratonovich integrals in the multidimensional\n      scheme. Stratonovich integrals are an alternative to Ito integrals, and\n      can be used interchangeably when defining the higher order terms in the\n      update equation. We use Stratonovich integrals here because they have a\n      convenient approximation scheme for calculating cross terms involving\n      different components of the Wiener process. See Eq. 8.10 in Section 5.8 of\n      [2]. Default value: `5`.\n    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.\n      Default value: None which means that the dtype implied by `times` is used.\n    name: Python string. The name to give this op.\n      Default value: `None` which maps to `milstein_sample`.\n  "
    name = name or 'milstein_sample'
    with tf.name_scope(name):
        if stratonovich_order <= 0:
            raise ValueError('`stratonovich_order` must be a positive integer.')
        times = tf.convert_to_tensor(times, dtype=dtype)
        if dtype is None:
            dtype = times.dtype
        if initial_state is None:
            initial_state = tf.zeros(dim, dtype=dtype)
        initial_state = tf.convert_to_tensor(initial_state, dtype=dtype, name='initial_state')
        num_requested_times = tff_utils.get_shape(times)[0]
        if num_time_steps is not None and time_step is not None:
            raise ValueError('Only one of either `num_time_steps` or `time_step` should be defined but not both')
        if time_step is None:
            if num_time_steps is None:
                raise ValueError('Either `num_time_steps` or `time_step` should be defined.')
            num_time_steps = tf.convert_to_tensor(num_time_steps, dtype=tf.int32, name='num_time_steps')
            time_step = times[-1] / tf.cast(num_time_steps, dtype=dtype)
        else:
            time_step = tf.convert_to_tensor(time_step, dtype=dtype, name='time_step')
        (times, keep_mask, time_indices) = utils.prepare_grid(times=times, time_step=time_step, num_time_steps=num_time_steps, dtype=dtype)
        if watch_params is not None:
            watch_params = [tf.convert_to_tensor(param, dtype=dtype) for param in watch_params]
        if grad_volatility_fn is None:

            def _grad_volatility_fn(current_time, current_state, input_gradients):
                if False:
                    print('Hello World!')
                return gradient.fwd_gradient(functools.partial(volatility_fn, current_time), current_state, input_gradients=input_gradients, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            grad_volatility_fn = _grad_volatility_fn
        input_gradients = None
        if dim > 1:
            input_gradients = tf.unstack(tf.eye(dim, dtype=dtype))
            input_gradients = [tf.broadcast_to(start, [num_samples, dim]) for start in input_gradients]
        return _sample(dim=dim, drift_fn=drift_fn, volatility_fn=volatility_fn, grad_volatility_fn=grad_volatility_fn, times=times, time_step=time_step, keep_mask=keep_mask, num_requested_times=num_requested_times, num_samples=num_samples, initial_state=initial_state, random_type=random_type, seed=seed, swap_memory=swap_memory, skip=skip, precompute_normal_draws=precompute_normal_draws, watch_params=watch_params, time_indices=time_indices, input_gradients=input_gradients, stratonovich_order=stratonovich_order, dtype=dtype)

def _sample(*, dim, drift_fn, volatility_fn, grad_volatility_fn, times, time_step, keep_mask, num_requested_times, num_samples, initial_state, random_type, seed, swap_memory, skip, precompute_normal_draws, watch_params, time_indices, input_gradients, stratonovich_order, dtype):
    if False:
        i = 10
        return i + 15
    'Returns a sample of paths from the process using the Milstein method.'
    dt = times[1:] - times[:-1]
    sqrt_dt = tf.sqrt(dt)
    current_state = initial_state + tf.zeros([num_samples, dim], dtype=initial_state.dtype)
    if dt.shape.is_fully_defined():
        steps_num = dt.shape.as_list()[-1]
    else:
        steps_num = tf.shape(dt)[-1]
    if precompute_normal_draws or random_type in (random.RandomType.SOBOL, random.RandomType.HALTON, random.RandomType.HALTON_RANDOMIZED, random.RandomType.STATELESS, random.RandomType.STATELESS_ANTITHETIC):
        all_normal_draws = utils.generate_mc_normal_draws(num_normal_draws=dim + 3 * dim * stratonovich_order, num_time_steps=steps_num, num_sample_paths=num_samples, random_type=random_type, dtype=dtype, seed=seed, skip=skip)
        normal_draws = all_normal_draws[:, :, :dim]
        wiener_mean = None
        aux_normal_draws = []
        start = dim
        for _ in range(3):
            end = start + dim * stratonovich_order
            aux_normal_draws.append(all_normal_draws[:, :, start:end])
            start = end
    else:
        wiener_mean = tf.zeros((dim,), dtype=dtype, name='wiener_mean')
        normal_draws = None
        aux_normal_draws = None
    if watch_params is None:
        return _while_loop(dim=dim, steps_num=steps_num, current_state=current_state, drift_fn=drift_fn, volatility_fn=volatility_fn, grad_volatility_fn=grad_volatility_fn, wiener_mean=wiener_mean, num_samples=num_samples, times=times, dt=dt, sqrt_dt=sqrt_dt, time_step=time_step, keep_mask=keep_mask, num_requested_times=num_requested_times, swap_memory=swap_memory, random_type=random_type, seed=seed, normal_draws=normal_draws, input_gradients=input_gradients, stratonovich_order=stratonovich_order, aux_normal_draws=aux_normal_draws, dtype=dtype)
    else:
        return _for_loop(dim=dim, steps_num=steps_num, current_state=current_state, drift_fn=drift_fn, volatility_fn=volatility_fn, grad_volatility_fn=grad_volatility_fn, wiener_mean=wiener_mean, num_samples=num_samples, times=times, dt=dt, sqrt_dt=sqrt_dt, time_indices=time_indices, keep_mask=keep_mask, watch_params=watch_params, random_type=random_type, seed=seed, normal_draws=normal_draws, input_gradients=input_gradients, stratonovich_order=stratonovich_order, aux_normal_draws=aux_normal_draws)

def _while_loop(*, dim, steps_num, current_state, drift_fn, volatility_fn, grad_volatility_fn, wiener_mean, num_samples, times, dt, sqrt_dt, time_step, num_requested_times, keep_mask, swap_memory, random_type, seed, normal_draws, input_gradients, stratonovich_order, aux_normal_draws, dtype):
    if False:
        while True:
            i = 10
    'Sample paths using tf.while_loop.'
    written_count = 0
    if isinstance(num_requested_times, int) and num_requested_times == 1:
        record_samples = False
        result = current_state
    else:
        record_samples = True
        element_shape = current_state.shape
        result = tf.TensorArray(dtype=dtype, size=num_requested_times, element_shape=element_shape, clear_after_read=False)
        result = result.write(written_count, current_state)
    written_count += tf.cast(keep_mask[0], dtype=tf.int32)

    def cond_fn(i, written_count, *args):
        if False:
            while True:
                i = 10
        del args
        return tf.math.logical_and(i < steps_num, written_count < num_requested_times)

    def step_fn(i, written_count, current_state, result):
        if False:
            while True:
                i = 10
        return _milstein_step(dim=dim, i=i, written_count=written_count, current_state=current_state, result=result, drift_fn=drift_fn, volatility_fn=volatility_fn, grad_volatility_fn=grad_volatility_fn, wiener_mean=wiener_mean, num_samples=num_samples, times=times, dt=dt, sqrt_dt=sqrt_dt, keep_mask=keep_mask, random_type=random_type, seed=seed, normal_draws=normal_draws, input_gradients=input_gradients, stratonovich_order=stratonovich_order, aux_normal_draws=aux_normal_draws, record_samples=record_samples)
    maximum_iterations = tf.cast(1.0 / time_step, dtype=tf.int32) + tf.size(times)
    (_, _, _, result) = tf.while_loop(cond_fn, step_fn, (0, written_count, current_state, result), maximum_iterations=maximum_iterations, swap_memory=swap_memory)
    if not record_samples:
        return tf.expand_dims(result, axis=-2)
    result = result.stack()
    n = result.shape.rank
    perm = list(range(1, n - 1)) + [0, n - 1]
    return tf.transpose(result, perm)

def _for_loop(*, dim, steps_num, current_state, drift_fn, volatility_fn, grad_volatility_fn, wiener_mean, watch_params, num_samples, times, dt, sqrt_dt, time_indices, keep_mask, random_type, seed, normal_draws, input_gradients, stratonovich_order, aux_normal_draws):
    if False:
        for i in range(10):
            print('nop')
    'Sample paths using custom for_loop.'
    num_time_points = time_indices.shape.as_list()[-1]
    if num_time_points == 1:
        iter_nums = steps_num
    else:
        iter_nums = time_indices

    def step_fn(i, current_state):
        if False:
            for i in range(10):
                print('nop')
        current_state = current_state[0]
        (_, _, next_state, _) = _milstein_step(dim=dim, i=i, written_count=0, current_state=current_state, result=tf.expand_dims(current_state, axis=1), drift_fn=drift_fn, volatility_fn=volatility_fn, grad_volatility_fn=grad_volatility_fn, wiener_mean=wiener_mean, num_samples=num_samples, times=times, dt=dt, sqrt_dt=sqrt_dt, keep_mask=keep_mask, random_type=random_type, seed=seed, normal_draws=normal_draws, input_gradients=input_gradients, stratonovich_order=stratonovich_order, aux_normal_draws=aux_normal_draws, record_samples=False)
        return [next_state]
    result = custom_loops.for_loop(body_fn=step_fn, initial_state=[current_state], params=watch_params, num_iterations=iter_nums)[0]
    if num_time_points == 1:
        return tf.expand_dims(result, axis=1)
    return tf.transpose(result, (1, 0, 2))

def _outer_prod(v1, v2):
    if False:
        print('Hello World!')
    'Computes the outer product of v1 and v2.'
    return tf.linalg.einsum('...i,...j->...ij', v1, v2)

def _stratonovich_integral(dim, dt, sqrt_dt, dw, stratonovich_draws, order):
    if False:
        return 10
    'Approximate Stratonovich integrals J(i, j).\n\n\n\n  Args:\n    dim: An integer. The dimension of the state.\n    dt: A double. The time step.\n    sqrt_dt: A double. The square root of dt.\n    dw: A double. The Wiener increment.\n    stratonovich_draws: A list of tensors corresponding to the independent\n      N(0,1) random variables used in the approximation.\n    order: An integer. The stratonovich_order.\n\n  Returns:\n    A Tensor of shape [dw.shape[0], dim, dim] corresponding to the Stratonovich\n    integral for each pairwise component of the Wiener process. In other words,\n    J(i,j) corresponds to an integral over W_i and W_j.\n  '
    p = order - 1
    sqrt_rho_p = tf.sqrt(tf.constant(1 / 12 - sum([1 / r ** 2 for r in range(1, order + 1)]) / 2 / _PI ** 2, dtype=dw.dtype))
    mu = stratonovich_draws[0]
    zeta = tf.transpose(stratonovich_draws[1], [2, 0, 1])
    eta = tf.transpose(stratonovich_draws[2], [2, 0, 1])
    xi = dw / sqrt_dt
    r_i = tf.stack([tf.ones(zeta[0, ...].shape + [dim], dtype=zeta.dtype) / r for r in range(1, order + 1)], 0)
    value = dt * (_outer_prod(dw, dw) / 2 + sqrt_rho_p * (_outer_prod(mu[..., p], xi) - _outer_prod(xi, mu[..., p])))
    value += dt * tf.reduce_sum(tf.multiply(_outer_prod(zeta, _SQRT_2 * xi + eta) - _outer_prod(_SQRT_2 * xi + eta, zeta), r_i), 0) / (2 * _PI)
    return value

def _milstein_hot(dim, vol, grad_vol, dt, sqrt_dt, dw, stratonovich_draws, stratonovich_order):
    if False:
        return 10
    'Higher order terms for Milstein update.'
    offdiag = _stratonovich_integral(dim=dim, dt=dt, sqrt_dt=sqrt_dt, dw=dw, stratonovich_draws=stratonovich_draws, order=stratonovich_order)
    stratonovich_integrals = tf.linalg.set_diag(offdiag, dw * dw / 2)
    stacked_grad_vol = []
    for state_ix in range(dim):
        stacked_grad_vol.append(tf.transpose(tf.stack([x[..., state_ix, :] for x in grad_vol], -1), [0, 2, 1]))
    stacked_grad_vol = tf.stack(stacked_grad_vol, 0)
    lbar = tf.matmul(stacked_grad_vol, vol)
    return tf.transpose(tf.reduce_sum(tf.multiply(lbar, stratonovich_integrals), [-2, -1]))

def _stratonovich_drift_update(num_samples, vol, grad_vol):
    if False:
        while True:
            i = 10
    'Updates drift function for use with stratonovich integrals.'
    vol = tf.reshape(vol, [num_samples, -1])
    grad_vol = tf.concat(grad_vol, 2)
    return tf.linalg.matvec(grad_vol, vol)

def _milstein_1d(dw, dt, sqrt_dt, current_state, drift, vol, grad_vol):
    if False:
        for i in range(10):
            print('nop')
    'Performs the milstein update in one dimension.'
    dw = dw * sqrt_dt
    dt_inc = dt * drift
    dw_inc = tf.linalg.matvec(vol, dw)
    hot_vol = tf.squeeze(tf.multiply(vol, grad_vol), -1)
    hot_dw = dw * dw - dt
    hot_inc = tf.multiply(hot_vol, hot_dw) / 2
    return current_state + dt_inc + dw_inc + hot_inc

def _milstein_nd(dim, num_samples, dw, dt, sqrt_dt, current_state, drift, vol, grad_vol, stratonovich_draws, stratonovich_order):
    if False:
        i = 10
        return i + 15
    'Performs the milstein update in multiple dimensions.'
    dw = dw * sqrt_dt
    drift_update = _stratonovich_drift_update(num_samples, vol, grad_vol)
    dt_inc = dt * (drift - drift_update)
    dw_inc = tf.linalg.matvec(vol, dw)
    hot_inc = _milstein_hot(dim=dim, vol=vol, grad_vol=grad_vol, dt=dt, sqrt_dt=sqrt_dt, dw=dw, stratonovich_draws=stratonovich_draws, stratonovich_order=stratonovich_order)
    return current_state + dt_inc + dw_inc + hot_inc

def _milstein_step(*, dim, i, written_count, current_state, result, drift_fn, volatility_fn, grad_volatility_fn, wiener_mean, num_samples, times, dt, sqrt_dt, keep_mask, random_type, seed, normal_draws, input_gradients, stratonovich_order, aux_normal_draws, record_samples):
    if False:
        return 10
    'Performs one step of Milstein scheme.'
    current_time = times[i + 1]
    written_count = tf.cast(written_count, tf.int32)
    if normal_draws is not None:
        dw = normal_draws[i]
    else:
        dw = random.mv_normal_sample((num_samples,), mean=wiener_mean, random_type=random_type, seed=seed)
    if aux_normal_draws is not None:
        stratonovich_draws = []
        for j in range(3):
            stratonovich_draws.append(tf.reshape(aux_normal_draws[j][i], [num_samples, dim, stratonovich_order]))
    else:
        stratonovich_draws = []
        for j in range(3):
            stratonovich_draws.append(random.mv_normal_sample((num_samples,), mean=tf.zeros((dim, stratonovich_order), dtype=current_state.dtype, name='stratonovich_draws_{}'.format(j)), random_type=random_type, seed=seed))
    if dim == 1:
        drift = drift_fn(current_time, current_state)
        vol = volatility_fn(current_time, current_state)
        grad_vol = grad_volatility_fn(current_time, current_state, tf.ones_like(current_state))
        next_state = _milstein_1d(dw=dw, dt=dt[i], sqrt_dt=sqrt_dt[i], current_state=current_state, drift=drift, vol=vol, grad_vol=grad_vol)
    else:
        drift = drift_fn(current_time, current_state)
        vol = volatility_fn(current_time, current_state)
        grad_vol = [grad_volatility_fn(current_time, current_state, start) for start in input_gradients]
        next_state = _milstein_nd(dim=dim, num_samples=num_samples, dw=dw, dt=dt[i], sqrt_dt=sqrt_dt[i], current_state=current_state, drift=drift, vol=vol, grad_vol=grad_vol, stratonovich_draws=stratonovich_draws, stratonovich_order=stratonovich_order)
    if record_samples:
        result = result.write(written_count, next_state)
    else:
        result = next_state
    written_count += tf.cast(keep_mask[i + 1], dtype=tf.int32)
    return (i + 1, written_count, next_state, result)
__all__ = ['sample']