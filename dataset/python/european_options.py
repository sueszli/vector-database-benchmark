# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sabr Approximations to European Option prices."""

import tensorflow.compat.v2 as tf

from tf_quant_finance.black_scholes import vanilla_prices
from tf_quant_finance.models.sabr.approximations.implied_volatility import implied_volatility
from tf_quant_finance.models.sabr.approximations.implied_volatility import SabrApproximationType
from tf_quant_finance.models.sabr.approximations.implied_volatility import SabrImpliedVolatilityType


def option_price(*,
                 strikes,
                 expiries,
                 forwards,
                 is_call_options,
                 alpha,
                 beta,
                 volvol,
                 rho,
                 shift=0.0,
                 volatility_type=SabrImpliedVolatilityType.LOGNORMAL,
                 approximation_type=SabrApproximationType.HAGAN,
                 dtype=None,
                 name=None):
  """Computes the approximate European option price under the SABR model.

  For a review of the SABR model and the conventions used, please see the
  docstring for `implied_volatility`.

  #### Example
  ```python
  import tf_quant_finance as tff
  import tensorflow.compat.v2 as tf

  prices = tff.models.sabr.approximations.european_option_price(
    strikes=np.array([90.0, 100.0]),
    expiries=np.array([0.5, 1.0]),
    forwards=np.array([100.0, 110.0]),
    is_call_options=np.array([True, False]),
    alpha=3.2,
    beta=0.2,
    volvol=1.4,
    rho=0.0005,
    dtype=tf.float64)

  # Expected: [10.41244961, 1.47123225]

  ```

  Args:
    strikes: Real `Tensor` of arbitrary shape, specifying the strike prices.
      Values must be strictly positive.
    expiries: Real `Tensor` of shape compatible with that of `strikes`,
      specifying the corresponding time-to-expiries of the options. Values must
      be strictly positive.
    forwards: Real `Tensor` of shape compatible with that of `strikes`,
      specifying the observed forward prices of the underlying. Values must be
      strictly positive.
    is_call_options: Boolean `Tensor` of shape compatible with that of
      `forward`, indicating whether the option is a call option (true) or put
      option (false).
    alpha: Real `Tensor` of shape compatible with that of `strikes`, specifying
      the initial values of the stochastic volatility. Values must be strictly
      positive.
    beta: Real `Tensor` of shape compatible with that of `strikes`, specifying
      the model exponent `beta`. Values must satisfy 0 <= `beta` <= 1.
    volvol: Real `Tensor` of shape compatible with that of `strikes`,
      specifying the model vol-vol multipliers. Values must satisfy
      `0 <= volvol`.
    rho: Real `Tensor` of shape compatible with that of `strikes`, specifying
      the correlation factors between the Wiener processes modeling the forward
      and the volatility. Values must satisfy -1 < `rho` < 1.
    shift: Optional `Tensor` of shape compatible with that of `strkies`,
      specifying the shift parameter(s). In the shifted model, the process
      modeling the forward is modified as: dF = sigma * (F + shift) ^ beta * dW.
      With this modification, negative forward rates are valid as long as
      F > -shift.
      Default value: 0.0
    volatility_type: Either SabrImpliedVolatility.NORMAL or LOGNORMAL.
      Default value: `LOGNORMAL`.
    approximation_type: Instance of `SabrApproxmationScheme`.
      Default value: `HAGAN`.
    dtype: Optional: `tf.DType`. If supplied, the dtype to be used for
      converting values to `Tensor`s.
      Default value: `None`, which means that the default dtypes inferred from
        `strikes` is used.
    name: str. The name for the ops created by this function.
      Default value: 'sabr_approx_eu_option_price'.

  Returns:
    A real `Tensor` of the same shape as `strikes`, containing the
    corresponding options price.
  """
  name = name or 'sabr_approx_eu_option_price'

  with tf.name_scope(name):
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    dtype = dtype or forwards.dtype

    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')

    is_call_options = tf.convert_to_tensor(
        is_call_options, dtype=tf.bool, name='is_call_options')

    if volatility_type == SabrImpliedVolatilityType.NORMAL:
      sigma_normal = implied_volatility(
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          alpha=alpha,
          beta=beta,
          volvol=volvol,
          rho=rho,
          shift=shift,
          volatility_type=volatility_type,
          approximation_type=approximation_type,
          dtype=dtype)

      return vanilla_prices.option_price(
          volatilities=sigma_normal,
          strikes=strikes + shift,
          expiries=expiries,
          forwards=forwards + shift,
          is_call_options=is_call_options,
          is_normal_volatility=True)

    elif volatility_type == SabrImpliedVolatilityType.LOGNORMAL:
      sigma_black = implied_volatility(
          strikes=strikes,
          expiries=expiries,
          forwards=forwards,
          alpha=alpha,
          beta=beta,
          volvol=volvol,
          rho=rho,
          shift=shift,
          volatility_type=volatility_type,
          approximation_type=approximation_type,
          dtype=dtype)

      return vanilla_prices.option_price(
          volatilities=sigma_black,
          strikes=strikes + shift,
          expiries=expiries,
          forwards=forwards + shift,
          is_call_options=is_call_options,
          is_normal_volatility=False)
