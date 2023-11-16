"""Analytic approximation for European option prices using SABR model."""
import enum
import tensorflow.compat.v2 as tf

@enum.unique
class SabrApproximationType(enum.Enum):
    """Approximation to the SABR model.

  * `HAGAN`: Using the Hagan approximation [1].

  #### References
  [1] Hagan et al, Managing Smile Risk, Wilmott (2002), 1:84-108
  """
    HAGAN = 1

@enum.unique
class SabrImpliedVolatilityType(enum.Enum):
    """The implied volality arising from the SABR approximate solution.

  * `NORMAL`: The volatility for the normal model, i.e. the `sigma_n` for a
    stochastic model of the underlying `F` behaving like:

    ```
    dF = sigma_n dW
    ```

  * `LOGNORMAL`: The volatility for the lognomal (aka Black) model, i.e. the
    `sigma_B` for a stochastic model of the underlying `F` behaving like:

    ```
    dF = sigma_b F dW
    ```

  """
    NORMAL = 1
    LOGNORMAL = 2

def implied_volatility(*, strikes, expiries, forwards, alpha, beta, volvol, rho, shift=0.0, volatility_type=SabrImpliedVolatilityType.LOGNORMAL, approximation_type=SabrApproximationType.HAGAN, dtype=None, name=None):
    if False:
        print('Hello World!')
    "Computes the implied volatility under the SABR model.\n\n  The SABR model specifies the risk neutral dynamics of the underlying as the\n  following set of stochastic differential equations:\n\n  ```\n    dF = sigma F^beta dW_1\n    dsigma = volvol sigma dW_2\n    dW1 dW2 = rho dt\n\n    F(0) = f\n    sigma(0) = alpha\n  ```\n  where F(t) represents the value of the forward price as a function of time,\n  and sigma(t) is the volatility.\n\n  Here, we implement an approximate solution as proposed by Hagan [1], and back\n  out the equivalent implied volatility that would've been obtained under either\n  the normal model or the Black model.\n\n  #### Example\n  ```python\n  import tf_quant_finance as tff\n  import tensorflow.compat.v2 as tf\n\n  equiv_vol = tff.models.sabr.approximations.implied_volatility(\n      strikes=np.array([106.0, 11.0]),\n      expiries=np.array([17.0 / 365.0, 400.0 / 365.0]),\n      forwards=np.array([120.0, 20.0]),\n      alpha=1.63,\n      beta=0.6,\n      rho=0.00002,\n      volvol=3.3,\n      dtype=tf.float64)\n  # Expected: [0.33284656705268817, 1.9828728139982792]\n\n  # Running this inside a unit test passes:\n  # equiv_vol = self.evaluate(equiv_vol)\n  # self.assertAllClose(equiv_vol, 0.33284656705268817)\n  ```\n  #### References\n  [1] Hagan et al, Managing Smile Risk, Wilmott (2002), 1:84-108\n\n  Args:\n    strikes: Real `Tensor` of arbitrary shape, specifying the strike prices.\n      Values must be strictly positive.\n    expiries: Real `Tensor` of shape compatible with that of `strikes`,\n      specifying the corresponding time-to-expiries of the options. Values must\n      be strictly positive.\n    forwards: Real `Tensor` of shape compatible with that of `strikes`,\n      specifying the observed forward prices of the underlying. Values must be\n      strictly positive.\n    alpha: Real `Tensor` of shape compatible with that of `strikes`, specifying\n      the initial values of the stochastic volatility. Values must be strictly\n      positive.\n    beta: Real `Tensor` of shape compatible with that of `strikes`, specifying\n      the model exponent `beta`. Values must satisfy 0 <= `beta` <= 1.\n    volvol: Real `Tensor` of shape compatible with that of `strikes`,\n      specifying the model vol-vol multipliers. Values of `volvol` must be\n      non-negative.\n    rho: Real `Tensor` of shape compatible with that of `strikes`, specifying\n      the correlation factors between the Wiener processes modeling the forward\n      and the volatility. Values must satisfy -1 < `rho` < 1.\n    shift: Optional `Tensor` of shape compatible with that of `strkies`,\n      specifying the shift parameter(s). In the shifted model, the process\n      modeling the forward is modified as: dF = sigma * (F + shift) ^ beta * dW.\n      With this modification, negative forward rates are valid as long as\n      F > -shift.\n      Default value: 0.0\n    volatility_type: Either SabrImpliedVolatility.NORMAL or LOGNORMAL.\n      Default value: `LOGNORMAL`.\n    approximation_type: Instance of `SabrApproxmationScheme`.\n      Default value: `HAGAN`.\n    dtype: Optional: `tf.DType`. If supplied, the dtype to be used for\n      converting values to `Tensor`s.\n      Default value: `None`, which means that the default dtypes inferred from\n        `strikes` is used.\n    name: str. The name for the ops created by this function.\n      Default value: 'sabr_approx_implied_volatility'.\n\n  Returns:\n    A real `Tensor` of the same shape as `strikes`, containing the\n    corresponding equivalent implied volatilities.\n  "
    name = name or 'sabr_approx_implied_volatility'
    del approximation_type
    with tf.name_scope(name):
        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        dtype = dtype or strikes.dtype
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
        alpha = tf.convert_to_tensor(alpha, dtype=dtype, name='alpha')
        beta = tf.convert_to_tensor(beta, dtype=dtype, name='beta')
        rho = tf.convert_to_tensor(rho, dtype=dtype, name='rho')
        volvol = tf.convert_to_tensor(volvol, dtype=dtype, name='volvol')
        strikes += shift
        forwards += shift
        moneyness = forwards / strikes
        log_moneyness = tf.math.log(moneyness)
        adj_moneyness = tf.math.pow(moneyness, 1.0 - beta)
        sqrt_adj_moneyness = tf.math.sqrt(adj_moneyness)
        adj_alpha = alpha * tf.math.pow(strikes, beta - 1.0)
        zeta = volvol / adj_alpha * sqrt_adj_moneyness * log_moneyness
        zeta_by_xhat = _zeta_by_xhat(zeta, rho, dtype)
        denom = _denom(beta, log_moneyness)
        correction_2 = rho * beta / 4.0 * (1.0 / sqrt_adj_moneyness) * (adj_alpha * volvol * expiries)
        correction_3 = (2.0 - 3.0 * rho * rho) / 24.0 * (volvol * volvol * expiries)
        if volatility_type == SabrImpliedVolatilityType.NORMAL:
            correction_1 = -beta * (2.0 - beta) / 24.0 * (1.0 / adj_moneyness) * (adj_alpha * adj_alpha * expiries)
            number = _denom(0.0, log_moneyness)
            return adj_alpha * strikes * tf.math.pow(moneyness, beta / 2.0) * (number / denom) * zeta_by_xhat * (1 + correction_1 + correction_2 + correction_3)
        elif volatility_type == SabrImpliedVolatilityType.LOGNORMAL:
            correction_1 = (1.0 - beta) * (1.0 - beta) / 24.0 * (1.0 / adj_moneyness) * (adj_alpha * adj_alpha * expiries)
            return adj_alpha * (1.0 / sqrt_adj_moneyness) * (1.0 / denom) * zeta_by_xhat * (1.0 + correction_1 + correction_2 + correction_3)
        else:
            raise ValueError('Invalid value of `volatility_type`')

def _epsilon(dtype):
    if False:
        while True:
            i = 10
    dtype = tf.as_dtype(dtype).as_numpy_dtype
    eps = 1e-06 if dtype == tf.float32.as_numpy_dtype else 1e-10
    return eps

def _zeta_by_xhat(zeta, rho, dtype):
    if False:
        print('Hello World!')
    zbxh = tf.math.divide_no_nan(zeta, tf.math.log((tf.math.sqrt(1 - 2 * rho * zeta + zeta * zeta) - rho + zeta) / (1.0 - rho)))
    eps = _epsilon(dtype)
    return tf.where(tf.abs(zeta) > eps, zbxh, 1.0)

def _denom(beta, log_f_by_k):
    if False:
        while True:
            i = 10
    s = (1.0 - beta) * log_f_by_k
    s_squared = s * s
    return 1.0 + s_squared / 24.0 + s_squared * s_squared / 1920.0