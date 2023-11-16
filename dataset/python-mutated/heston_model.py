"""Heston model with piecewise constant parameters."""
from typing import Union, Callable, Optional
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.math import piecewise
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.models import generic_ito_process
from tf_quant_finance.models import utils
__all__ = ['HestonModel']
_SQRT_2 = np.sqrt(2.0, dtype=np.float64)
_CallableOrTensor = Union[Callable, types.RealTensor]

class HestonModel(generic_ito_process.GenericItoProcess):
    """Heston Model with piecewise constant parameters.

  Represents the Ito process:

  ```None
    dX(t) = -V(t) / 2 * dt + sqrt(V(t)) * dW_{X}(t),
    dV(t) = mean_reversion(t) * (theta(t) - V(t)) * dt
            + volvol(t) * sqrt(V(t)) * dW_{V}(t)
  ```

  where `W_{X}` and `W_{V}` are 1D Brownian motions with a correlation
  `rho(t)`. `mean_reversion`, `theta`, `volvol`, and `rho` are positive
  piecewise constant functions of time. Here `V(t)` represents the process
  variance at time `t` and `X` represents logarithm of the spot price at time
  `t`.

  `mean_reversion` corresponds to the mean reversion rate, `theta` is the long
  run price variance, and `volvol` is the volatility of the volatility.

  See [1] and [2] for details.

  #### Example

  ```python
  import tf_quant_finance as tff
  import numpy as np
  volvol = tff.math.piecewise.PiecewiseConstantFunc(
      jump_locations=[0.5], values=[1, 1.1], dtype=np.float64)
  process = tff.models.HestonModel(
      mean_reversion=0.5, theta=0.04, volvol=volvol, rho=0.1, dtype=np.float64)
  times = np.linspace(0.0, 1.0, 1000)
  num_samples = 10000  # number of trajectories
  sample_paths = process.sample_paths(
      times,
      time_step=0.01,
      num_samples=num_samples,
      initial_state=np.array([1.0, 0.04]),
      random_type=random.RandomType.SOBOL)
  ```

  #### References:
    [1]: Cristian Homescu. Implied volatility surface: construction
      methodologies and characteristics.
      arXiv: https://arxiv.org/pdf/1107.1834.pdf
    [2]: Leif Andersen. Efficient Simulation of the Heston Stochastic
      Volatility Models. 2006.
      Link:
      http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/d512ad5b22d73cc1c1257052003f1aed/1826b88b152e65a7c12574b000347c74/$FILE/LeifAndersenHeston.pdf
  """

    def __init__(self, mean_reversion: _CallableOrTensor, theta: _CallableOrTensor, volvol: _CallableOrTensor, rho: _CallableOrTensor, dtype: Optional[tf.DType]=None, name: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        'Initializes the Heston Model.\n\n    #### References:\n      [1]: Leif Andersen. Efficient Simulation of the Heston Stochastic\n        Volatility Models. 2006.\n        Link:\n        http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/d512ad5b22d73cc1c1257052003f1aed/1826b88b152e65a7c12574b000347c74/$FILE/LeifAndersenHeston.pdf\n    Args:\n      mean_reversion: Scalar real `Tensor` or an instant of batch-free\n        left-continuous `PiecewiseConstantFunc`. Should contain a positive\n        value.\n        Corresponds to the mean reversion rate.\n      theta: Scalar real `Tensor` or an instant of batch-free left-continuous\n        `PiecewiseConstantFunc`. Should contain positive a value of the same\n        `dtype` as `mean_reversion`.\n        Corresponds to the long run price variance.\n      volvol: Scalar real `Tensor` or an instant of batch-free left-continuous\n        `PiecewiseConstantFunc`. Should contain positive a value of the same\n        `dtype` as `mean_reversion`.\n        Corresponds to the volatility of the volatility.\n      rho: Scalar real `Tensor` or an instant of batch-free left-continuous\n        `PiecewiseConstantFunc`. Should contain a value in range (-1, 1) of the\n        same `dtype` as `mean_reversion`.\n        Corresponds to the correlation between dW_{X}` and `dW_{V}`.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: `None` which maps to `tf.float32`.\n      name: Python string. The name to give to the ops created by this class.\n        Default value: `None` which maps to the default name `heston_model`.\n    '
        self._name = name or 'heston_model'
        with tf.name_scope(self._name):
            self._dtype = dtype or tf.float32
            if isinstance(mean_reversion, piecewise.PiecewiseConstantFunc):
                self._mean_reversion = mean_reversion
            else:
                self._mean_reversion = tf.convert_to_tensor(mean_reversion, dtype=self._dtype, name='mean_reversion')
            self._theta = theta if isinstance(theta, piecewise.PiecewiseConstantFunc) else tf.convert_to_tensor(theta, dtype=self._dtype, name='theta')
            self._volvol = volvol if isinstance(volvol, piecewise.PiecewiseConstantFunc) else tf.convert_to_tensor(volvol, dtype=self._dtype, name='volvol')
            self._rho = rho if isinstance(rho, piecewise.PiecewiseConstantFunc) else tf.convert_to_tensor(rho, dtype=self._dtype, name='rho')

        def _vol_fn(t, x):
            if False:
                while True:
                    i = 10
            'Volatility function of the Heston Process.'
            vol = tf.sqrt(tf.abs(x[..., 1]))
            zeros = tf.zeros_like(vol)
            (rho, volvol) = _get_parameters([t], self._rho, self._volvol)
            (rho, volvol) = (rho[0], volvol[0])
            vol_matrix_1 = tf.stack([vol, volvol * rho * vol], -1)
            vol_matrix_2 = tf.stack([zeros, volvol * tf.sqrt(1 - rho ** 2) * vol], -1)
            vol_matrix = tf.stack([vol_matrix_1, vol_matrix_2], -1)
            return vol_matrix

        def _drift_fn(t, x):
            if False:
                i = 10
                return i + 15
            var = x[..., 1]
            (mean_reversion, theta) = _get_parameters([t], self._mean_reversion, self._theta)
            (mean_reversion, theta) = (mean_reversion[0], theta[0])
            log_spot_drift = -var / 2
            var_drift = mean_reversion * (theta - var)
            drift = tf.stack([log_spot_drift, var_drift], -1)
            return drift
        super(HestonModel, self).__init__(2, _drift_fn, _vol_fn, self._dtype, name)

    def sample_paths(self, times: types.RealTensor, initial_state: types.RealTensor, num_samples: types.IntTensor=1, random_type: Optional[random.RandomType]=None, seed: Optional[types.RealTensor]=None, time_step: Optional[types.RealTensor]=None, skip: types.IntTensor=0, tolerance: types.RealTensor=1e-06, num_time_steps: Optional[types.IntTensor]=None, precompute_normal_draws: types.BoolTensor=True, times_grid: Optional[types.RealTensor]=None, normal_draws: Optional[types.RealTensor]=None, name: Optional[str]=None) -> types.RealTensor:
        if False:
            return 10
        "Returns a sample of paths from the process.\n\n    Using Quadratic-Exponential (QE) method described in [1] generates samples\n    paths started at time zero and returns paths values at the specified time\n    points.\n\n    Args:\n      times: Rank 1 `Tensor` of positive real values. The times at which the\n        path points are to be evaluated.\n      initial_state: A rank 1 `Tensor` with two elements where the first element\n        corresponds to the initial value of the log spot `X(0)` and the second\n        to the starting variance value `V(0)`.\n      num_samples: Positive scalar `int`. The number of paths to draw.\n      random_type: Enum value of `RandomType`. The type of (quasi)-random\n        number generator to use to generate the paths.\n        Default value: None which maps to the standard pseudo-random numbers.\n      seed: Seed for the random number generator. The seed is\n        only relevant if `random_type` is one of\n        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,\n          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and\n        `HALTON_RANDOMIZED` the seed should be an Python integer. For\n        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer\n        `Tensor` of shape `[2]`.\n        Default value: `None` which means no seed is set.\n      time_step: Positive Python float to denote time discretization parameter.\n      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n        Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.\n      tolerance: Scalar positive real `Tensor`. Specifies minimum time tolerance\n        for which the stochastic process `X(t) != X(t + tolerance)`.\n        Default value: 1e-6.\n      num_time_steps: An optional Scalar integer `Tensor` - a total number of\n        time steps performed by the algorithm. The maximal distance between\n        points in grid is bounded by\n        `times[-1] / (num_time_steps - times.shape[0])`.\n        Either this or `time_step` should be supplied.\n        Default value: `None`.\n      precompute_normal_draws: Python bool. Indicates whether the noise\n        increments `N(0, t_{n+1}) - N(0, t_n)` are precomputed. For `HALTON`\n        and `SOBOL` random types the increments are always precomputed. While\n        the resulting graph consumes more memory, the performance gains might\n        be significant.\n        Default value: `True`.\n      times_grid: An optional rank 1 `Tensor` representing time discretization\n        grid. If `times` are not on the grid, then the nearest points from the\n        grid are used. When supplied, `num_time_steps` and `time_step` are\n        ignored.\n        Default value: `None`, which means that times grid is computed using\n        `time_step` and `num_time_steps`.\n      normal_draws: A `Tensor` of shape broadcastable with\n        `[num_samples, num_time_points, 2]` and the same\n        `dtype` as `times`. Represents random normal draws to compute increments\n        `N(0, t_{n+1}) - N(0, t_n)`. When supplied, `num_samples` argument is\n        ignored and the first dimensions of `normal_draws` is used instead.\n        Default value: `None` which means that the draws are generated by the\n        algorithm. By default normal_draws for each model in the batch are\n        independent.\n      name: Str. The name to give this op.\n        Default value: `sample_paths`.\n\n    Returns:\n      A `Tensor`s of shape [num_samples, k, 2] where `k` is the size\n      of the `times`. For each sample and time the first dimension represents\n      the simulated log-state trajectories of the spot price `X(t)`, whereas the\n      second one represents the simulated variance trajectories `V(t)`.\n\n    Raises:\n      ValueError: If `time_step` is not supplied.\n\n    #### References:\n      [1]: Leif Andersen. Efficient Simulation of the Heston Stochastic\n        Volatility Models. 2006.\n    "
        if random_type is None:
            random_type = random.RandomType.PSEUDO
        name = name or self._name + '_sample_path'
        with tf.name_scope(name):
            times = tf.convert_to_tensor(times, self._dtype)
            if normal_draws is not None:
                normal_draws = tf.convert_to_tensor(normal_draws, dtype=self._dtype, name='normal_draws')
                perm = [1, 0, 2]
                normal_draws = tf.transpose(normal_draws, perm=perm)
                num_samples = tff_utils.get_shape(normal_draws)[-2]
            current_log_spot = tf.convert_to_tensor(initial_state[..., 0], dtype=self._dtype) + tf.zeros([num_samples], dtype=self._dtype)
            current_vol = tf.convert_to_tensor(initial_state[..., 1], dtype=self._dtype) + tf.zeros([num_samples], dtype=self._dtype)
            num_requested_times = tff_utils.get_shape(times)[0]
            if times_grid is None:
                if time_step is None:
                    if num_time_steps is None:
                        raise ValueError('When `times_grid` is not supplied, either `num_time_steps` or `time_step` should be defined.')
                    else:
                        num_time_steps = tf.convert_to_tensor(num_time_steps, dtype=tf.int32, name='num_time_steps')
                        time_step = times[-1] / tf.cast(num_time_steps, dtype=self._dtype)
                else:
                    if num_time_steps is not None:
                        raise ValueError('Both `time_step` and `num_time_steps` can not be `None` simultaneously when calling sample_paths of HestonModel.')
                    time_step = tf.convert_to_tensor(time_step, dtype=self._dtype, name='time_step')
            else:
                times_grid = tf.convert_to_tensor(times_grid, dtype=self._dtype, name='times_grid')
            (times, keep_mask) = _prepare_grid(times, time_step, times.dtype, self._mean_reversion, self._theta, self._volvol, self._rho, num_time_steps=num_time_steps, times_grid=times_grid)
            return self._sample_paths(times=times, num_requested_times=num_requested_times, current_log_spot=current_log_spot, current_vol=current_vol, num_samples=num_samples, random_type=random_type, keep_mask=keep_mask, seed=seed, skip=skip, tolerance=tolerance, precompute_normal_draws=precompute_normal_draws, normal_draws=normal_draws)

    def _sample_paths(self, times, num_requested_times, current_log_spot, current_vol, num_samples, random_type, keep_mask, seed, skip, tolerance, precompute_normal_draws, normal_draws):
        if False:
            return 10
        'Returns a sample of paths from the process.'
        dt = times[1:] - times[:-1]
        (mean_reversion, theta, volvol, rho) = _get_parameters(times + tf.reduce_min(dt) / 2, self._mean_reversion, self._theta, self._volvol, self._rho)
        if dt.shape.is_fully_defined():
            steps_num = dt.shape.as_list()[-1]
        else:
            steps_num = tf.shape(dt)[-1]
            if random_type == random.RandomType.SOBOL:
                raise ValueError('Sobol sequence for Euler sampling is temporarily unsupported when `time_step` or `times` have a non-constant value')
        if normal_draws is None:
            if precompute_normal_draws or random_type in (random.RandomType.SOBOL, random.RandomType.HALTON, random.RandomType.HALTON_RANDOMIZED, random.RandomType.STATELESS, random.RandomType.STATELESS_ANTITHETIC):
                normal_draws = utils.generate_mc_normal_draws(num_normal_draws=2, num_time_steps=steps_num, num_sample_paths=num_samples, random_type=random_type, dtype=self.dtype(), seed=seed, skip=skip)
            else:
                normal_draws = None
        written_count = 0
        if isinstance(num_requested_times, int) and num_requested_times == 1:
            record_samples = False
            log_spot_paths = current_log_spot
            vol_paths = current_vol
        else:
            record_samples = True
            element_shape = current_log_spot.shape
            log_spot_paths = tf.TensorArray(dtype=times.dtype, size=num_requested_times, element_shape=element_shape, clear_after_read=False)
            vol_paths = tf.TensorArray(dtype=times.dtype, size=num_requested_times, element_shape=element_shape, clear_after_read=False)
            log_spot_paths = log_spot_paths.write(written_count, current_log_spot)
            vol_paths = vol_paths.write(written_count, current_vol)
        written_count += tf.cast(keep_mask[0], dtype=tf.int32)

        def cond_fn(i, written_count, *args):
            if False:
                i = 10
                return i + 15
            del args
            return tf.math.logical_and(i < steps_num, written_count < num_requested_times)

        def body_fn(i, written_count, current_vol, current_log_spot, vol_paths, log_spot_paths):
            if False:
                print('Hello World!')
            'Simulate Heston process to the next time point.'
            time_step = dt[i]
            if normal_draws is None:
                normals = random.mv_normal_sample((num_samples,), mean=tf.zeros([2], dtype=mean_reversion.dtype), seed=seed)
            else:
                normals = normal_draws[i]

            def _next_vol_fn():
                if False:
                    i = 10
                    return i + 15
                return _update_variance(mean_reversion[i], theta[i], volvol[i], rho[i], current_vol, time_step, normals[..., 0])
            next_vol = tf.cond(time_step > tolerance, _next_vol_fn, lambda : current_vol)

            def _next_log_spot_fn():
                if False:
                    print('Hello World!')
                return _update_log_spot(mean_reversion[i], theta[i], volvol[i], rho[i], current_vol, next_vol, current_log_spot, time_step, normals[..., 1])
            next_log_spot = tf.cond(time_step > tolerance, _next_log_spot_fn, lambda : current_log_spot)
            if record_samples:
                vol_paths = vol_paths.write(written_count, next_vol)
                log_spot_paths = log_spot_paths.write(written_count, next_log_spot)
            else:
                vol_paths = next_vol
                log_spot_paths = next_log_spot
            written_count += tf.cast(keep_mask[i + 1], dtype=tf.int32)
            return (i + 1, written_count, next_vol, next_log_spot, vol_paths, log_spot_paths)
        (_, _, _, _, vol_paths, log_spot_paths) = tf.while_loop(cond_fn, body_fn, (0, 0, current_vol, current_log_spot, vol_paths, log_spot_paths), maximum_iterations=steps_num)
        if not record_samples:
            vol_paths = tf.expand_dims(vol_paths, axis=-1)
            log_spot_paths = tf.expand_dims(log_spot_paths, axis=-1)
            return tf.stack([log_spot_paths, vol_paths], -1)
        vol_paths = vol_paths.stack()
        log_spot_paths = log_spot_paths.stack()
        vol_paths = tf.transpose(vol_paths)
        log_spot_paths = tf.transpose(log_spot_paths)
        return tf.stack([log_spot_paths, vol_paths], -1)

    def expected_total_variance(self, future_times: types.RealTensor, initial_var: types.RealTensor, name: Optional[str]=None) -> types.RealTensor:
        if False:
            for i in range(10):
                print('nop')
        "Computes the expected variance of the process up to `future_time`.\n\n    The Heston model affords a closed form expression for its expected variance:\n\n    `E[S_T] = (V(0) - theta)(1 - e^{-mean_reversion * T})/ mean_reversion +\n    theta * T`\n\n    Where `S_T` represents the integral of the instantaneous variance process V\n    from 0 to `T` [p138. of 1].\n\n    ### References\n    [1] Gatheral, Jim. The volatility surface: a practitioner's guide. Vol. 357.\n      John Wiley & Sons, 2011.\n\n    Args:\n      future_times: real `Tensor` representing times in the future (`T` in the\n        above notation).\n      initial_var: real `Tensor` of shape compatible with `future_time`. The\n        value of the variance process at time zero, `V(0)`.\n      name: Python `str`. The name to give this op.\n        Default value: name of the instance + `_expected_total_variance`\n\n    Returns:\n      The expected variance at `future_time`.\n\n    Raises:\n      ValueError: for non-constant parameters.\n    "
        for param_name in ['_mean_reversion', '_theta']:
            param = getattr(self, param_name)
            if not isinstance(param, tf.Tensor):
                raise ValueError(f'Only constant values supported for {param_name}')
        name = name or self._name + '_expected_total_variance'
        with tf.name_scope(name):
            future_times = tf.convert_to_tensor(future_times, self._dtype, name='future_times')
            initial_var = tf.convert_to_tensor(initial_var, self._dtype, name='initial_var')
            reversion_strength = (1 - tf.math.exp(-self._mean_reversion * future_times)) / self._mean_reversion
            return (initial_var - self._theta) * reversion_strength + self._theta * future_times

def _get_parameters(times, *params):
    if False:
        print('Hello World!')
    'Gets parameter values at at specified `times`.'
    result = []
    for param in params:
        if isinstance(param, piecewise.PiecewiseConstantFunc):
            result.append(param(times))
        else:
            result.append(param * tf.ones_like(times))
    return result

def _update_variance(mean_reversion, theta, volvol, rho, current_vol, time_step, normals, psi_c=1.5):
    if False:
        for i in range(10):
            print('nop')
    'Updates variance value.'
    del rho
    psi_c = tf.convert_to_tensor(psi_c, dtype=mean_reversion.dtype)
    scaled_time = tf.exp(-mean_reversion * time_step)
    volvol_squared = volvol ** 2
    m = theta + (current_vol - theta) * scaled_time
    s_squared = current_vol * volvol_squared * scaled_time / mean_reversion * (1 - scaled_time) + theta * volvol_squared / 2 / mean_reversion * (1 - scaled_time) ** 2
    psi = s_squared / m ** 2
    uniforms = 0.5 * (1 + tf.math.erf(normals / _SQRT_2))
    cond = psi < psi_c
    psi_inv = 2 / psi
    b_squared = psi_inv - 1 + tf.sqrt(psi_inv * (psi_inv - 1))
    a = m / (1 + b_squared)
    next_var_true = a * (tf.sqrt(b_squared) + tf.squeeze(normals)) ** 2
    p = (psi - 1) / (psi + 1)
    beta = (1 - p) / m
    next_var_false = tf.where(uniforms > p, tf.math.log(1 - p) - tf.math.log(1 - uniforms), tf.zeros_like(uniforms)) / beta
    next_var = tf.where(cond, next_var_true, next_var_false)
    return next_var

def _update_log_spot(mean_reversion, theta, volvol, rho, current_vol, next_vol, current_log_spot, time_step, normals, gamma_1=0.5, gamma_2=0.5):
    if False:
        print('Hello World!')
    'Updates log-spot value.'
    k_0 = -rho * mean_reversion * theta / volvol * time_step
    k_1 = gamma_1 * time_step * (mean_reversion * rho / volvol - 0.5) - rho / volvol
    k_2 = gamma_2 * time_step * (mean_reversion * rho / volvol - 0.5) + rho / volvol
    k_3 = gamma_1 * time_step * (1 - rho ** 2)
    k_4 = gamma_2 * time_step * (1 - rho ** 2)
    next_log_spot = current_log_spot + k_0 + k_1 * current_vol + k_2 * next_vol + tf.sqrt(k_3 * current_vol + k_4 * next_vol) * normals
    return next_log_spot

def _prepare_grid(times, time_step, dtype, *params, num_time_steps=None, times_grid=None):
    if False:
        while True:
            i = 10
    "Prepares grid of times for path generation.\n\n  Args:\n    times:  Rank 1 `Tensor` of increasing positive real values. The times at\n      which the path points are to be evaluated.\n    time_step: Rank 0 real `Tensor`. Maximal distance between points in\n      resulting grid.\n    dtype: `tf.Dtype` of the input and output `Tensor`s.\n    *params: Parameters of the Heston model. Either scalar `Tensor`s of the\n      same `dtype` or instances of `PiecewiseConstantFunc`.\n    num_time_steps: Number of points on the grid. If suppied, a uniform grid\n      is constructed for `[time_step, times[-1] - time_step]` consisting of\n      max(0, num_time_steps - len(times)) points that is then concatenated with\n      times. This parameter guarantees the number of points on the time grid\n      is `max(len(times), num_time_steps)` and that `times` are included to the\n      grid.\n      Default value: `None`, which means that a uniform grid is created.\n      containing all points from 'times` and the uniform grid of points between\n      `[0, times[-1]]` with grid size equal to `time_step`.\n    times_grid: An optional rank 1 `Tensor` representing time discretization\n      grid. If `times` are not on the grid, then the nearest points from the\n      grid are used.\n      Default value: `None`, which means that times grid is computed using\n      `time_step` and `num_time_steps`.\n\n  Returns:\n    Tuple `(all_times, mask)`.\n    `all_times` is a 1-D real `Tensor` containing all points from 'times`, the\n    uniform grid of points between `[0, times[-1]]` with grid size equal to\n    `time_step`, and jump locations of piecewise constant parameters The\n    `Tensor` is sorted in ascending order and may contain duplicates.\n    `mask` is a boolean 1-D `Tensor` of the same shape as 'all_times', showing\n    which elements of 'all_times' correspond to THE values from `times`.\n    Guarantees that times[0]=0 and mask[0]=False.\n  "
    additional_times = []
    for param in params:
        if isinstance(param, piecewise.PiecewiseConstantFunc):
            additional_times.append(param.jump_locations())
    if times_grid is None:
        if time_step is not None:
            grid = tf.range(0.0, times[-1], time_step, dtype=dtype)
            all_times = tf.concat([grid, times] + additional_times, axis=0)
        elif num_time_steps is not None:
            grid = tf.linspace(tf.convert_to_tensor(0.0, dtype=dtype), times[-1] - time_step, num_time_steps)
            all_times = tf.concat([grid, times] + additional_times, axis=0)
        additional_times_mask = [tf.zeros_like(times, dtype=tf.bool) for times in additional_times]
        mask = tf.concat([tf.zeros_like(grid, dtype=tf.bool), tf.ones_like(times, dtype=tf.bool)] + additional_times_mask, axis=0)
        perm = tf.argsort(all_times, stable=True)
        all_times = tf.gather(all_times, perm)
        mask = tf.gather(mask, perm)
    else:
        (all_times, mask, _) = utils.prepare_grid(times=times, time_step=time_step, times_grid=times_grid, dtype=dtype)
    return (all_times, mask)