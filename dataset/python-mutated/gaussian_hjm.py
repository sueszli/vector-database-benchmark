"""Multi-Factor Gaussian HJM Model."""
from typing import Callable, Union
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.math import gradient
from tf_quant_finance.math import piecewise
from tf_quant_finance.math import random
from tf_quant_finance.models import euler_sampling
from tf_quant_finance.models import utils
from tf_quant_finance.models.hjm import quasi_gaussian_hjm
__all__ = ['GaussianHJM']

class GaussianHJM(quasi_gaussian_hjm.QuasiGaussianHJM):
    """Gaussian HJM model for term-structure modeling.

  Heath-Jarrow-Morton (HJM) model for the interest rate term-structre
  modelling specifies the dynamics of the instantaneus forward rate `f(t,T)`
  with maturity `T` at time `t` as follows:

  ```None
    df(t,T) = mu(t,T) dt + sum_i sigma_i(t,  T) * dW_i(t),
    1 <= i <= n,
  ```
  where `mu(t,T)` and `sigma_i(t,T)` denote the drift and volatility
  for the forward rate and `W_i` are Brownian motions with instantaneous
  correlation `Rho`. The model above represents an `n-factor` HJM model.
  The Gaussian HJM model assumes that the volatility `sigma_i(t,T)` is a
  deterministic function of time (t). Under the risk-neutral measure, the
  drift `mu(t,T)` is computed as

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
  matrix. For Gaussian HJM model, the quantity `y_ij(t)` can be computed
  analytically as follows:

  ```None
    y_ij(t) = rho_ij * exp(-k_i * t) * exp(-k_j * t) *
              int_0^t exp((k_i+k_j) * s) * sigma_i(s) * sigma_j(s) ds
  ```

  The Gaussian HJM class implements the model outlined above by simulating the
  state `x(t)` while analytically computing `y(t)`.

  The price at time `t` of a zero-coupon bond maturing at `T` is given by
  (Ref. [1]):

  ```None
  P(t,T) = P(0,T) / P(0,t) *
           exp(-x(t) * G(t,T) - 0.5 * y(t) * G(t,T)^2)
  ```

  The HJM model implementation supports constant mean-reversion rate `k` and
  `sigma(t)` can be an arbitrary function of `t`. We use Euler discretization
  to simulate the HJM model.

  #### Example. Simulate a 4-factor HJM process.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64
  def discount_fn(x):
    return 0.01 * tf.ones_like(x, dtype=dtype)

  process = tff.models.hjm.GaussianHJM(
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

    def __init__(self, dim: int, mean_reversion: types.RealTensor, volatility: Union[types.RealTensor, Callable[..., types.RealTensor]], initial_discount_rate_fn, corr_matrix: types.RealTensor=None, dtype: tf.DType=None, name: str=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the HJM model.\n\n    Args:\n      dim: A Python scalar which corresponds to the number of factors comprising\n        the model.\n      mean_reversion: A real positive `Tensor` of shape `[dim]`. Corresponds to\n        the mean reversion rate of each factor.\n      volatility: A real positive `Tensor` of the same `dtype` and shape as\n        `mean_reversion` or a callable with the following properties: (a)  The\n          callable should accept a scalar `Tensor` `t` and returns a 1-D\n          `Tensor` of shape `[dim]`. The function returns instantaneous\n          volatility `sigma(t)`. When `volatility` is specified is a real\n          `Tensor`, each factor is assumed to have a constant instantaneous\n          volatility. Corresponds to the instantaneous volatility of each\n          factor.\n      initial_discount_rate_fn: A Python callable that accepts expiry time as a\n        real `Tensor` of the same `dtype` as `mean_reversion` and returns a\n        `Tensor` of shape `input_shape`. Corresponds to the zero coupon bond\n        yield at the present time for the input expiry time.\n      corr_matrix: A `Tensor` of shape `[dim, dim]` and the same `dtype` as\n        `mean_reversion`. Corresponds to the correlation matrix `Rho`.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: `None` which maps to `tf.float32`.\n      name: Python string. The name to give to the ops created by this class.\n        Default value: `None` which maps to the default name\n          `gaussian_hjm_model`.\n    '
        self._name = name or 'gaussian_hjm_model'
        with tf.name_scope(self._name):
            self._dtype = dtype or tf.float32
            self._dim = dim
            self._factors = dim

            def _instant_forward_rate_fn(t):
                if False:
                    return 10
                t = tf.convert_to_tensor(t, dtype=self._dtype)

                def _log_zero_coupon_bond(x):
                    if False:
                        print('Hello World!')
                    r = tf.convert_to_tensor(initial_discount_rate_fn(x), dtype=self._dtype)
                    return -r * x
                rate = -gradient.fwd_gradient(_log_zero_coupon_bond, t, use_gradient_tape=True, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                return rate

            def _initial_discount_rate_fn(t):
                if False:
                    return 10
                return tf.convert_to_tensor(initial_discount_rate_fn(t), dtype=self._dtype)
            self._instant_forward_rate_fn = _instant_forward_rate_fn
            self._initial_discount_rate_fn = _initial_discount_rate_fn
            self._mean_reversion = tf.convert_to_tensor(mean_reversion, dtype=dtype, name='mean_reversion')
            self._batch_shape = []
            self._batch_rank = 0
            if callable(volatility):
                self._volatility = volatility
            else:
                volatility = tf.convert_to_tensor(volatility, dtype=dtype)
                jump_locations = [[]] * dim
                volatility = tf.expand_dims(volatility, axis=-1)
                self._volatility = piecewise.PiecewiseConstantFunc(jump_locations=jump_locations, values=volatility, dtype=dtype)
            if corr_matrix is None:
                corr_matrix = tf.eye(dim, dim, dtype=self._dtype)
            self._rho = tf.convert_to_tensor(corr_matrix, dtype=dtype, name='rho')
            self._sqrt_rho = tf.linalg.cholesky(self._rho)

            def _vol_fn(t, state):
                if False:
                    i = 10
                    return i + 15
                'Volatility function of Gaussian-HJM.'
                del state
                volatility = self._volatility(tf.expand_dims(t, -1))
                return self._sqrt_rho * volatility

            def _drift_fn(t, state):
                if False:
                    return 10
                'Drift function of Gaussian-HJM.'
                x = state
                y = self.state_y(tf.expand_dims(t, axis=-1))[..., 0]
                drift = tf.math.reduce_sum(y, axis=-1) - self._mean_reversion * x
                return drift
            self._exact_discretization_setup(dim)
            super(quasi_gaussian_hjm.QuasiGaussianHJM, self).__init__(dim, _drift_fn, _vol_fn, self._dtype, self._name)

    def sample_paths(self, times: types.RealTensor, num_samples: types.IntTensor, time_step: types.RealTensor=None, num_time_steps: types.IntTensor=None, random_type: random.RandomType=None, seed: types.IntTensor=None, skip: types.IntTensor=0, name: str=None) -> types.RealTensor:
        if False:
            print('Hello World!')
        "Returns a sample of short rate paths from the HJM process.\n\n    Uses Euler sampling for simulating the short rate paths.\n\n    Args:\n      times: A real positive `Tensor` of shape `(num_times,)`. The times at\n        which the path points are to be evaluated.\n      num_samples: Positive scalar `int32` `Tensor`. The number of paths to\n        draw.\n      time_step: Scalar real `Tensor`. Maximal distance between time grid points\n        in Euler scheme. Used only when Euler scheme is applied.\n        Default value: `None`.\n      num_time_steps: An optional Scalar integer `Tensor` - a total number of\n        time steps performed by the algorithm. The maximal distance between\n        points in grid is bounded by\n        `times[-1] / (num_time_steps - times.shape[0])`.\n        Either this or `time_step` should be supplied.\n        Default value: `None`.\n      random_type: Enum value of `RandomType`. The type of (quasi)-random\n        number generator to use to generate the paths.\n        Default value: `None` which maps to the standard pseudo-random numbers.\n      seed: Seed for the random number generator. The seed is\n        only relevant if `random_type` is one of\n        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,\n          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and\n        `HALTON_RANDOMIZED` the seed should be an Python integer. For\n        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer\n        `Tensor` of shape `[2]`.\n        Default value: `None` which means no seed is set.\n      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n        Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.\n        Default value: `0`.\n      name: Python string. The name to give this op.\n        Default value: `sample_paths`.\n\n    Returns:\n      A tuple containing four elements.\n\n      * The first element is a `Tensor` of\n      shape `[num_samples, num_times]` containing the simulated short rate\n      paths.\n      * The second element is a `Tensor` of shape\n      `[num_samples, num_times]` containing the simulated discount factor\n      paths.\n      * The third element is a `Tensor` of shape\n      `[num_samples, num_times, dim]` conating the simulated values of the\n      state variable `x`\n      * The fourth element is a `Tensor` of shape\n      `[num_samples, num_times, dim^2]` conating the simulated values of the\n      state variable `y`.\n\n    Raises:\n      ValueError:\n        (a) If `times` has rank different from `1`.\n        (b) If Euler scheme is used by times is not supplied.\n    "
        name = name or self._name + '_sample_path'
        with tf.name_scope(name):
            times = tf.convert_to_tensor(times, self._dtype)
            if times.shape.rank != 1:
                raise ValueError('`times` should be a rank 1 Tensor. Rank is {} instead.'.format(times.shape.rank))
            return self._sample_paths(times, time_step, num_time_steps, num_samples, random_type, skip, seed)

    def state_y(self, t: types.RealTensor, name: str=None) -> types.RealTensor:
        if False:
            i = 10
            return i + 15
        'Computes the state variable `y(t)` for tha Gaussian HJM Model.\n\n    For Gaussian HJM model, the state parameter y(t), can be analytically\n    computed as follows:\n\n    y_ij(t) = exp(-k_i * t) * exp(-k_j * t) * (\n              int_0^t rho_ij * sigma_i(u) * sigma_j(u) * du)\n\n    Args:\n      t: A rank 1 real `Tensor` of shape `[num_times]` specifying the time `t`.\n      name: Python string. The name to give to the ops created by this function.\n        Default value: `None` which maps to the default name `state_y`.\n\n    Returns:\n      A real `Tensor` of shape [self._factors, self._factors, num_times]\n      containing the computed y_ij(t).\n    '
        name = name or 'state_y'
        with tf.name_scope(name):
            t = tf.convert_to_tensor(t, dtype=self._dtype)
            t_shape = tf.shape(t)
            t = tf.broadcast_to(t, tf.concat([[self._dim], t_shape], axis=0))
            time_index = tf.searchsorted(self._jump_locations, t)
            mr2 = tf.expand_dims(self._mean_reversion, axis=-1)
            mr2 = tf.expand_dims(mr2 + tf.transpose(mr2), axis=-1)

            def _integrate_volatility_squared(vol, l_limit, u_limit):
                if False:
                    print('Hello World!')
                vol = tf.expand_dims(vol, axis=-2)
                vol_squared = tf.expand_dims(self._rho, axis=-1) * (vol * tf.transpose(vol, perm=[1, 0, 2]))
                return vol_squared / mr2 * (tf.math.exp(mr2 * u_limit) - tf.math.exp(mr2 * l_limit))
            is_constant_vol = tf.math.equal(tf.shape(self._jump_values_vol)[-1], 0)
            v_squared_between_vol_knots = tf.cond(is_constant_vol, lambda : tf.zeros(shape=(self._dim, self._dim, 0), dtype=self._dtype), lambda : _integrate_volatility_squared(self._jump_values_vol, self._padded_knots, self._jump_locations))
            v_squared_at_vol_knots = tf.concat([tf.zeros((self._dim, self._dim, 1), dtype=self._dtype), utils.cumsum_using_matvec(v_squared_between_vol_knots)], axis=-1)
            vn = tf.concat([self._zero_padding, self._jump_locations], axis=1)
            v_squared_t = _integrate_volatility_squared(self._volatility(t), tf.gather(vn, time_index, batch_dims=1), t)
            v_squared_t += tf.gather(v_squared_at_vol_knots, time_index, batch_dims=-1)
            return tf.math.exp(-mr2 * t) * v_squared_t

    def discount_bond_price(self, state: types.RealTensor, times: types.RealTensor, maturities: types.RealTensor, name: str=None) -> types.RealTensor:
        if False:
            return 10
        'Returns zero-coupon bond prices `P(t,T)` conditional on `x(t)`.\n\n    Args:\n      state: A `Tensor` of real dtype and shape compatible with\n        `(num_times, dim)` specifying the state `x(t)`.\n      times: A `Tensor` of real dtype and shape `(num_times,)`. The time `t`\n        at which discount bond prices are computed.\n      maturities: A `Tensor` of real dtype and shape `(num_times,)`. The time\n        to maturity of the discount bonds.\n      name: Str. The name to give this op.\n        Default value: `discount_bond_prices`.\n\n    Returns:\n      A `Tensor` of real dtype and the same shape as `(num_times,)`\n      containing the price of zero-coupon bonds.\n    '
        name = name or self._name + '_discount_bond_prices'
        with tf.name_scope(name):
            x_t = tf.convert_to_tensor(state, self._dtype)
            times = tf.convert_to_tensor(times, self._dtype)
            maturities = tf.convert_to_tensor(maturities, self._dtype)
            input_shape_times = tf.shape(times)
            mean_reversion = self._mean_reversion
            y_t = self.state_y(times)
            y_t = tf.reshape(tf.transpose(y_t), tf.concat([input_shape_times, [self._dim, self._dim]], axis=0))
            values = self._bond_reconstitution(times, maturities, mean_reversion, x_t, y_t, 1, tf.shape(times)[0])
            return values[0][0]

    def _sample_paths(self, times, time_step, num_time_steps, num_samples, random_type, skip, seed):
        if False:
            for i in range(10):
                print('nop')
        'Returns a sample of paths from the process.'
        initial_state = tf.zeros((self._dim,), dtype=self._dtype)
        time_step_internal = time_step
        if num_time_steps is not None:
            num_time_steps = tf.convert_to_tensor(num_time_steps, dtype=tf.int32, name='num_time_steps')
            time_step_internal = times[-1] / tf.cast(num_time_steps, dtype=self._dtype)
        (times, _, time_indices) = utils.prepare_grid(times=times, time_step=time_step_internal, dtype=self._dtype, num_time_steps=num_time_steps)
        dt = times[1:] - times[:-1]
        paths = euler_sampling.sample(self._dim, self._drift_fn, self._volatility_fn, times, num_time_steps=num_time_steps, num_samples=num_samples, initial_state=initial_state, random_type=random_type, seed=seed, time_step=time_step, skip=skip)
        y_paths = self.state_y(times)
        y_paths = tf.reshape(y_paths, tf.concat([[self._dim ** 2], tf.shape(times)], axis=0))
        y_paths = tf.repeat(tf.expand_dims(tf.transpose(y_paths), axis=0), num_samples, axis=0)
        f_0_t = self._instant_forward_rate_fn(times)
        rate_paths = tf.math.reduce_sum(paths, axis=-1) + f_0_t
        discount_factor_paths = tf.math.exp(-rate_paths[:, :-1] * dt)
        discount_factor_paths = tf.concat([tf.ones((num_samples, 1), dtype=self._dtype), discount_factor_paths], axis=1)
        discount_factor_paths = utils.cumprod_using_matvec(discount_factor_paths)
        return (tf.gather(rate_paths, time_indices, axis=1), tf.gather(discount_factor_paths, time_indices, axis=1), tf.gather(paths, time_indices, axis=1), tf.gather(y_paths, time_indices, axis=1))

    def _exact_discretization_setup(self, dim):
        if False:
            for i in range(10):
                print('nop')
        'Initial setup for efficient computations.'
        self._zero_padding = tf.zeros((dim, 1), dtype=self._dtype)
        self._jump_locations = self._volatility.jump_locations()
        self._jump_values_vol = self._volatility(self._jump_locations)
        self._padded_knots = tf.concat([self._zero_padding, self._jump_locations[:, :-1]], axis=1)