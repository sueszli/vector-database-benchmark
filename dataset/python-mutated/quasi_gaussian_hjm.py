"""Multi-Factor Quasi-Gaussian HJM Model."""
from typing import Callable, Union, Tuple
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.math import gradient
from tf_quant_finance.math import random
from tf_quant_finance.models import euler_sampling
from tf_quant_finance.models import generic_ito_process
from tf_quant_finance.models import utils
__all__ = ['QuasiGaussianHJM']

class QuasiGaussianHJM(generic_ito_process.GenericItoProcess):
    """Quasi-Gaussian HJM model for term-structure modeling.

  Heath-Jarrow-Morton (HJM) model for the interest rate term-structre
  modelling specifies the dynamics of the instantaneus forward rate `f(t,T)`
  with maturity `T` at time `t` as follows:

  ```None
    df(t,T) = mu(t,T) dt + sum_i sigma_i(t,  T) * dW_i(t),
    1 <= i <= n,
  ```
  where `mu(t,T)` and `sigma_i(t,T)` denote the drift and volatility
  for the forward rate and `W_i` are Brownian motions with instantaneous
  correlation `Rho`. The model above represents an `n-factor` HJM model. Under
  the risk-neutral measure, the drift `mu(t,T)` is computed as

  ```
    mu(t,T) = sum_i sigma_i(t,T)  int_t^T sigma_(t,u) du
  ```
  Using the separability condition, the HJM model above can be formulated as
  the following Markovian model:

  ```None
    sigma(t,T) = sigma(t) * h(T)    (Separability condition)
  ```
  A common choice for the function h(t) is `h(t) = exp(-kt)`. Using the above
  parameterization of sigma(t,T), we obtain the following Markovian
  formulation of the HJM model [1]:

  ```None
    HJM Model
    dx_i(t) = (sum_j [y_ij(t)] - k_i * x_i(t)) dt + sigma_i(t) dW_i
    dy_ij(t) = (rho_ij * sigma_i(t)*sigma_j(t) - (k_i + k_j) * y_ij(t)) dt
    r(t) = sum_i x_i(t) + f(0, t)
  ```
  where `x` is an `n`-dimensional vector and `y` is an `nxn` dimensional
  matrix. The HJM class implements the model outlined above by jointly
  simulating the state [x_t, y_t].

  The price at time `t` of a zero-coupon bond maturing at `T` is given by
  (Ref. [1]):

  ```None
  P(t,T) = P(0,T) / P(0,t) *
           exp(-x(t) * G(t,T) - 0.5 * y(t) * G(t,T)^2)
  ```

  The HJM model implentation supports constant mean-reversion rate `k` and
  `sigma(t)` can be an arbitrary function of `t` and `r`. The implementation
  uses Euler discretization to simulate the HJM model.

  #### Example. Simulate a 4-factor HJM process.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64
  def discount_fn(x):
    return 0.01 * tf.ones_like(x, dtype=dtype)

  process = tff.models.hjm.QuasiGaussianHJM(
      dim=4,
      mean_reversion=[0.03, 0.01, 0.02, 0.005],  # constant mean-reversion
      volatility=[0.01, 0.011, 0.015, 0.008],  # constant volatility
      initial_discount_rate_fn=discount_fn,
      dtype=dtype)
  times = np.array([0.1, 1.0, 2.0, 3.0])
  short_rate_paths, discount_paths, _, _ = process.sample_paths(
      times,
      num_samples=100000,
      time_step=0.1,
      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
      seed=[1, 2],
      skip=1000000)
  ```

  #### References:
    [1]: Leif B. G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling.
    Volume II: Term Structure Models.
  """

    def __init__(self, dim: int, mean_reversion: types.RealTensor, volatility: Union[types.RealTensor, Callable[..., types.RealTensor]], initial_discount_rate_fn: Callable[..., types.RealTensor], corr_matrix: types.RealTensor=None, validate_args: bool=False, dtype: tf.DType=None, name: str=None):
        if False:
            while True:
                i = 10
        'Initializes a batch of HJM models.\n\n    Args:\n      dim: A Python scalar which corresponds to the number of factors\n        comprising the model.\n      mean_reversion: A real positive `Tensor` of shape `batch_shape + [dim]`.\n        `batch_shape` denotes the shape of independent HJM models within the\n        batch. Corresponds to the mean reversion rate of each factor.\n      volatility: A real positive `Tensor` of the same `dtype` and shape as\n        `mean_reversion` or a callable with the following properties:\n        (a)  The callable should accept a scalar `Tensor` `t` and a `Tensor`\n        `r(t)` of shape `batch_shape + [num_samples]` and returns a `Tensor` of\n        shape compatible with `batch_shape + [num_samples, dim]`. The variable\n        `t`  stands for time and `r(t)` is the short rate at time `t`. The\n        function returns instantaneous volatility `sigma(t) = sigma(t, r(t))`.\n        When `volatility` is specified as a real `Tensor`, each factor is\n        assumed to have a constant instantaneous volatility  and the  model is\n        effectively a Gaussian HJM model.\n        Corresponds to the instantaneous volatility of each factor.\n      initial_discount_rate_fn: A Python callable that accepts expiry time as\n        a real `Tensor` of the same `dtype` as `mean_reversion` and returns a\n        `Tensor` of shape `batch_shape + input_shape`.\n        Corresponds to the zero coupon bond yield at the present time for the\n        input expiry time.\n      corr_matrix: A `Tensor` of shape `batch_shape + [dim, dim]` and the same\n        `dtype` as `mean_reversion`.\n        Corresponds to the correlation matrix `Rho`.\n      validate_args: Optional boolean flag to enable validation of the input\n        correlation matrix. If the flag is enabled and the input correlation\n        matrix is not positive semidefinite, an error is raised.\n        Default value: False.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: `None` which maps to `tf.float32`.\n      name: Python string. The name to give to the ops created by this class.\n        Default value: `None` which maps to the default name\n        `quasi_gaussian_hjm_model`.\n    '
        self._name = name or 'quasi_gaussian_hjm_model'
        with tf.name_scope(self._name):
            self._dtype = dtype or tf.float32
            self._dim = dim + dim ** 2
            self._factors = dim

            def _instant_forward_rate_fn(t):
                if False:
                    i = 10
                    return i + 15
                t = tf.convert_to_tensor(t, dtype=self._dtype)

                def _log_zero_coupon_bond(x):
                    if False:
                        while True:
                            i = 10
                    r = tf.convert_to_tensor(initial_discount_rate_fn(x), dtype=self._dtype)
                    return -r * x
                rate = -gradient.fwd_gradient(_log_zero_coupon_bond, t, use_gradient_tape=True, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                return rate

            def _initial_discount_rate_fn(t):
                if False:
                    print('Hello World!')
                return tf.convert_to_tensor(initial_discount_rate_fn(t), dtype=self._dtype)
            self._instant_forward_rate_fn = _instant_forward_rate_fn
            self._initial_discount_rate_fn = _initial_discount_rate_fn
            mean_reversion = tf.convert_to_tensor(mean_reversion, dtype=dtype, name='mean_reversion')

            def _infer_batch_shape():
                if False:
                    print('Hello World!')
                zero = tf.constant([0], dtype=self._dtype)
                return _initial_discount_rate_fn(zero).shape.as_list()[:-1]
            self._batch_shape = _infer_batch_shape()
            self._batch_rank = len(self._batch_shape)
            self._mean_reversion = mean_reversion
            if callable(volatility):
                self._volatility = volatility
            else:
                volatility = tf.convert_to_tensor(volatility, dtype=dtype)
                if self._batch_rank > 0:
                    volatility = tf.expand_dims(volatility, axis=self._batch_rank)

                def _tensor_to_volatility_fn(t, r):
                    if False:
                        print('Hello World!')
                    del t, r
                    return volatility
                self._volatility = _tensor_to_volatility_fn
            if corr_matrix is None:
                corr_matrix = tf.eye(dim, dim, batch_shape=self._batch_shape, dtype=self._dtype)
            self._rho = tf.convert_to_tensor(corr_matrix, dtype=self._dtype, name='rho')
            if validate_args:
                try:
                    self._sqrt_rho = tf.linalg.cholesky(self._rho)
                except:
                    raise ValueError('The input correlation matrix is not positive semidefinite.')
            else:
                self._sqrt_rho = _get_valid_sqrt_matrix(self._rho)

        def _vol_fn(t, state):
            if False:
                for i in range(10):
                    print('nop')
            'Volatility function of qG-HJM.'
            x = state[..., :self._factors]
            batch_shape_x = x.shape.as_list()[:-1]
            r_t = self._instant_forward_rate_fn(t) + tf.reduce_sum(x, axis=-1, keepdims=True)
            volatility = self._volatility(t, r_t)
            volatility = tf.expand_dims(volatility, axis=-1)
            diffusion_x = tf.broadcast_to(tf.expand_dims(self._sqrt_rho, axis=self._batch_rank) * volatility, batch_shape_x + [self._factors, self._factors])
            paddings = tf.constant([[0, 0]] * len(batch_shape_x) + [[0, self._factors ** 2], [0, self._factors ** 2]], dtype=tf.int32)
            diffusion = tf.pad(diffusion_x, paddings)
            return diffusion

        def _drift_fn(t, state):
            if False:
                print('Hello World!')
            'Drift function of qG-HJM.'
            x = state[..., :self._factors]
            y = state[..., self._factors:]
            batch_shape_x = x.shape.as_list()[:-1]
            y = tf.reshape(y, batch_shape_x + [self._factors, self._factors])
            r_t = self._instant_forward_rate_fn(t) + tf.reduce_sum(x, axis=-1, keepdims=True)
            volatility = self._volatility(t, r_t)
            volatility = tf.expand_dims(volatility, axis=-1)
            volatility_squared = tf.linalg.matmul(volatility, volatility, transpose_b=True)
            mr2 = tf.expand_dims(self._mean_reversion, axis=-1)
            perm = list(range(self._batch_rank)) + [self._batch_rank + 1, self._batch_rank]
            mr2 = mr2 + tf.transpose(mr2, perm=perm)
            mr2 = tf.expand_dims(mr2, axis=self._batch_rank)
            mr = self._mean_reversion
            if self._batch_rank > 0:
                mr = tf.expand_dims(self._mean_reversion, axis=1)
            drift_x = tf.math.reduce_sum(y, axis=-1) - mr * x
            drift_y = tf.expand_dims(self._rho, axis=self._batch_rank) * volatility_squared - mr2 * y
            drift_y = tf.reshape(drift_y, batch_shape_x + [self._factors * self._factors])
            drift = tf.concat([drift_x, drift_y], axis=-1)
            return drift
        super(QuasiGaussianHJM, self).__init__(self._dim, _drift_fn, _vol_fn, self._dtype, self._name)

    def sample_paths(self, times: types.RealTensor, num_samples: types.IntTensor, time_step: types.RealTensor, num_time_steps: types.IntTensor=None, random_type: random.RandomType=None, seed: types.IntTensor=None, skip: types.IntTensor=0, name: str=None) -> types.RealTensor:
        if False:
            i = 10
            return i + 15
        "Returns a sample of short rate paths from the HJM process.\n\n    Uses Euler sampling for simulating the short rate paths. The code closely\n    follows the notations in [1], section ###.\n\n    Args:\n      times: A real positive `Tensor` of shape `(num_times,)`. The times at\n        which the path points are to be evaluated.\n      num_samples: Positive scalar `int32` `Tensor`. The number of paths to\n        draw.\n      time_step: Scalar real `Tensor`. Maximal distance between time grid points\n        in Euler scheme. Used only when Euler scheme is applied.\n        Default value: `None`.\n      num_time_steps: An optional Scalar integer `Tensor` - a total number of\n        time steps performed by the algorithm. The maximal distance between\n        points in grid is bounded by\n        `times[-1] / (num_time_steps - times.shape[0])`.\n        Either this or `time_step` should be supplied.\n        Default value: `None`.\n      random_type: Enum value of `RandomType`. The type of (quasi)-random\n        number generator to use to generate the paths.\n        Default value: `None` which maps to the standard pseudo-random numbers.\n      seed: Seed for the random number generator. The seed is\n        only relevant if `random_type` is one of\n        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,\n          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and\n        `HALTON_RANDOMIZED` the seed should be an Python integer. For\n        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer\n        `Tensor` of shape `[2]`.\n        Default value: `None` which means no seed is set.\n      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n        Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.\n        Default value: `0`.\n      name: Python string. The name to give this op.\n        Default value: `sample_paths`.\n\n    Returns:\n      A tuple containing four elements.\n\n      * The first element is a `Tensor` of\n      shape `batch_shape + [num_samples, num_times]` containing the simulated\n      short rate paths.\n      * The second element is a `Tensor` of shape\n      `batch_shape + [num_samples, num_times]` containing the simulated\n      discount factor paths.\n      * The third element is a `Tensor` of shape\n      `batch_shape + [num_samples, num_times, dim]` conating the simulated\n      values of the state variable `x`\n      * The fourth element is a `Tensor` of shape\n      `batch_shape + [num_samples, num_times, dim^2]` conating the simulated\n      values of the state variable `y`.\n\n    Raises:\n      ValueError:\n        (a) If `times` has rank different from `1`.\n        (b) If Euler scheme is used by times is not supplied.\n    "
        name = name or self._name + '_sample_path'
        with tf.name_scope(name):
            times = tf.convert_to_tensor(times, self._dtype)
            if len(times.shape) != 1:
                raise ValueError('`times` should be a rank 1 Tensor. Rank is {} instead.'.format(len(times.shape)))
            return self._sample_paths(times, time_step, num_time_steps, num_samples, random_type, skip, seed)

    def sample_discount_curve_paths(self, times: types.RealTensor, curve_times: types.RealTensor, num_samples: types.IntTensor, time_step: types.RealTensor, num_time_steps: types.IntTensor=None, random_type: random.RandomType=None, seed: types.IntTensor=None, skip: types.IntTensor=0, name: str=None) -> Tuple[types.RealTensor, types.RealTensor, types.RealTensor]:
        if False:
            print('Hello World!')
        "Returns a sample of simulated discount curves for the Hull-white model.\n\n    Args:\n      times: A real positive `Tensor` of shape `[num_times,]`. The times `t` at\n        which the discount curves are to be evaluated.\n      curve_times: A real positive `Tensor` of shape `[num_curve_times]`. The\n        maturities at which discount curve is computed at each simulation time.\n      num_samples: Positive scalar `int`. The number of paths to draw.\n      time_step: Scalar real `Tensor`. Maximal distance between time grid points\n        in Euler scheme. Used only when Euler scheme is applied.\n        Default value: `None`.\n      num_time_steps: An optional Scalar integer `Tensor` - a total number of\n        time steps performed by the algorithm. The maximal distance betwen\n        points in grid is bounded by\n        `times[-1] / (num_time_steps - times.shape[0])`.\n        Either this or `time_step` should be supplied.\n        Default value: `None`.\n      random_type: Enum value of `RandomType`. The type of (quasi)-random\n        number generator to use to generate the paths.\n        Default value: None which maps to the standard pseudo-random numbers.\n      seed: Seed for the random number generator. The seed is\n        only relevant if `random_type` is one of\n        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,\n          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and\n        `HALTON_RANDOMIZED` the seed should be an Python integer. For\n        `STATELESS` and  `STATELESS_ANTITHETIC` must be supplied as an integer\n        `Tensor` of shape `[2]`.\n        Default value: `None` which means no seed is set.\n      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n        Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.\n        Default value: `0`.\n      name: Str. The name to give this op.\n        Default value: `sample_discount_curve_paths`.\n\n    Returns:\n      A tuple containing three `Tensor`s.\n\n      * The first element is a `Tensor` of shape\n      `batch_shape + [num_samples, num_curve_times, num_times]` containing\n      the simulated zero coupon bond curves `P(t, T)`.\n      * The second element is a `Tensor` of shape\n      `batch_shape + [num_samples, num_times]` containing the simulated short\n      rate paths.\n      * The third element is a `Tensor` of shape\n      `batch_shape + [num_samples, num_times]` containing the simulated\n      discount factor paths.\n\n    ### References:\n      [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,\n      Volume II: Term Structure Models. 2010.\n    "
        name = name or self._name + '_sample_discount_curve_paths'
        with tf.name_scope(name):
            times = tf.convert_to_tensor(times, self._dtype)
            num_times = tf.shape(times)[0]
            curve_times = tf.convert_to_tensor(curve_times, self._dtype)
            (rate_paths, discount_factor_paths, x_t, y_t) = self._sample_paths(times, time_step, num_time_steps, num_samples, random_type, skip, seed)
            x_t = tf.expand_dims(x_t, axis=self._batch_rank + 1)
            y_t = tf.expand_dims(y_t, axis=self._batch_rank + 1)
            num_curve_nodes = tf.shape(curve_times)[0]
            num_sim_steps = tf.shape(times)[0]
            times = tf.reshape(times, (1, 1, num_sim_steps))
            curve_times = tf.reshape(curve_times, (1, num_curve_nodes, 1))
            mean_reversion = tf.reshape(self._mean_reversion, self._batch_shape + [1, 1, 1, self._factors])
            return (self._bond_reconstitution(times, times + curve_times, mean_reversion, x_t, y_t, num_samples, num_times), rate_paths, discount_factor_paths)

    def _sample_paths(self, times, time_step, num_time_steps, num_samples, random_type, skip, seed):
        if False:
            print('Hello World!')
        'Returns a sample of paths from the process.'
        initial_state = tf.zeros(self._batch_shape + [1, self._dim], dtype=self._dtype)
        time_step_internal = time_step
        if num_time_steps is not None:
            num_time_steps = tf.convert_to_tensor(num_time_steps, dtype=tf.int32, name='num_time_steps')
            time_step_internal = times[-1] / tf.cast(num_time_steps, dtype=self._dtype)
        (times, _, time_indices) = utils.prepare_grid(times=times, time_step=time_step_internal, dtype=self._dtype, num_time_steps=num_time_steps)
        dt = times[1:] - times[:-1]
        xy_paths = euler_sampling.sample(self._dim, self._drift_fn, self._volatility_fn, times, num_samples=num_samples, initial_state=initial_state, random_type=random_type, seed=seed, time_step=time_step, num_time_steps=num_time_steps, skip=skip)
        x_paths = xy_paths[..., :self._factors]
        y_paths = xy_paths[..., self._factors:]
        f_0_t = self._instant_forward_rate_fn(times)
        rate_paths = tf.math.reduce_sum(x_paths, axis=-1) + tf.expand_dims(f_0_t, axis=-2)
        dt = tf.concat([tf.convert_to_tensor([0.0], dtype=self._dtype), dt], axis=0)
        discount_factor_paths = tf.math.exp(-utils.cumsum_using_matvec(rate_paths * dt))
        return (tf.gather(rate_paths, time_indices, axis=-1), tf.gather(discount_factor_paths, time_indices, axis=-1), tf.gather(x_paths, time_indices, axis=self._batch_rank + 1), tf.gather(y_paths, time_indices, axis=self._batch_rank + 1))

    def _bond_reconstitution(self, times, maturities, mean_reversion, x_t, y_t, num_samples, num_times):
        if False:
            for i in range(10):
                print('nop')
        'Computes discount bond prices using Eq. 10.18 in Ref [2].'
        times_expand = tf.expand_dims(times, axis=-1)
        maturities_expand = tf.expand_dims(maturities, axis=-1)
        p_0_t = tf.math.exp(-self._initial_discount_rate_fn(times) * times)
        p_0_t_tau = tf.math.exp(-self._initial_discount_rate_fn(maturities) * maturities) / p_0_t
        g_t_tau = (1.0 - tf.math.exp(-mean_reversion * (maturities_expand - times_expand))) / mean_reversion
        term1 = tf.math.reduce_sum(x_t * g_t_tau, axis=-1)
        y_t = tf.reshape(y_t, self._batch_shape + [num_samples, 1, num_times, self._factors, self._factors])
        term2 = tf.math.reduce_sum(g_t_tau * tf.linalg.matvec(y_t, g_t_tau), axis=-1)
        p_t_tau = p_0_t_tau * tf.math.exp(-term1 - 0.5 * term2)
        return p_t_tau

def _get_valid_sqrt_matrix(rho):
    if False:
        for i in range(10):
            print('nop')
    'Returns a matrix L such that rho = LL^T.'
    (e, v) = tf.linalg.eigh(rho)

    def _psd_true():
        if False:
            while True:
                i = 10
        return tf.linalg.cholesky(rho)

    def _psd_false():
        if False:
            for i in range(10):
                print('nop')
        realv = tf.math.real(v)
        adjusted_e = tf.linalg.diag(tf.maximum(tf.math.real(e), 1e-05))
        return tf.matmul(realv, tf.math.sqrt(adjusted_e))
    return tf.cond(tf.math.reduce_any(tf.less(tf.math.real(e), 1e-05)), _psd_false, _psd_true)