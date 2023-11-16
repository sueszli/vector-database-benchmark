# Copyright 2020 Google LLC
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
"""Pricing of Interest rate Caps/Floors using Heath-Jarrow-Morton model."""

from typing import Callable, Union

import tensorflow.compat.v2 as tf

from tf_quant_finance import types
from tf_quant_finance.math import random
from tf_quant_finance.models.hjm import zero_coupon_bond_option as zcb

__all__ = ['cap_floor_price']


def cap_floor_price(
    *,
    strikes: types.RealTensor,
    expiries: types.RealTensor,
    maturities: types.RealTensor,
    daycount_fractions: types.RealTensor,
    reference_rate_fn: Callable[..., types.RealTensor],
    dim: int,
    mean_reversion: types.RealTensor,
    volatility: Union[types.RealTensor, Callable[..., types.RealTensor]],
    corr_matrix: types.RealTensor = None,
    notional: types.RealTensor = 1.0,
    is_cap: types.BoolTensor = True,
    num_samples: types.IntTensor = 1,
    random_type: random.RandomType = None,
    seed: types.IntTensor = None,
    skip: types.IntTensor = 0,
    time_step: types.RealTensor = None,
    dtype: tf.DType = None,
    name: str = None) -> types.RealTensor:
  """Calculates the prices of interest rate Caps/Floors using the HJM model.

  An interest Cap (or Floor) is a portfolio of call (or put) options where the
  underlying for the individual options are successive forward rates. The
  individual options comprising a Cap are called Caplets and the corresponding
  options comprising a Floor are called Floorlets. For example, a
  caplet on forward rate `F(T_i, T_{i+1})` has the following payoff at time
  `T_{i_1}`:

  ```None

   caplet payoff = tau_i * max[F(T_i, T_{i+1}) - X, 0]

  ```
  where where `X` is the strake rate and `tau_i` is the daycount fraction. The
  caplet payoff (at `T_{i+1}`) can be expressed as the following at `T_i`:

  ```None

  caplet_payoff = (1.0 + tau_i * X) *
                  max[1.0 / (1 + tau_i * X) - P(T_i, T_{t+1}), 0]

  ```

  where `P(T_i, T_{i+1})` is the price at `T_i` of a zero coupon bond with
  maturity `T_{i+1}. Thus, a caplet can be priced as a put option on zero
  coupon bond [1].

  #### References
    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.
    Second Edition. 2007.

  #### Example
  The example shows how value a batch containing spot starting 1-year and
  2-year Caps and with quarterly frequency.

  ````python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64

  reference_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)
  expiries = np.array([[0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
                       [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]])
  maturities = np.array([[0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]])
  strikes = 0.01 * np.ones_like(expiries)
  daycount_fractions = np.array([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
  price = tff.models.hjm.cap_floor_price(
      strikes=strikes,
      expiries=expiries,
      maturities=maturities,
      daycount_fractions=daycount_fractions,
      notional=1.0e6,
      dim=1,
      mean_reversion=[0.03],
      volatility=[0.02],
      reference_rate_fn=reference_rate_fn,
      num_samples=500000,
      time_step=0.025,
      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
      seed=[1, 2],
      dtype=dtype)
  # Expected value: [[4071.821182], [15518.53244292]]
  ````

  Args:
    strikes: A real `Tensor` of any shape and dtype. The strike rate of the
      caplets or floorlets. The shape of this input determines the number (and
      shape) of the options to be priced and the shape of the output. For an
      N-dimensional input `Tensor`, the first N-1 dimensions correspond to the
      batch dimension, i.e., the distinct caps and floors and the last dimension
      correspond to the caplets or floorlets contained with an instrument.
    expiries: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The reset time of each caplet (or floorlet).
    maturities: A real `Tensor` of the same dtype and compatible shape as
      `strikes`.  The maturity time of each caplet (or floorlet) and also the
      time at which payment is made.
    daycount_fractions: A real `Tensor` of the same dtype and compatible shape
      as `strikes`. The daycount fractions associated with the underlying
      forward rates.
    reference_rate_fn: A Python callable that accepts expiry time as a real
      `Tensor` and returns a `Tensor` of shape `input_shape`. Returns the
      continuously compounded zero rate at the present time for the input expiry
      time.
    dim: A Python scalar which corresponds to the number of factors within a
      single HJM model.
    mean_reversion: A real positive `Tensor` of shape `[dim]`. Corresponds to
      the mean reversion rate of each factor.
    volatility: A real positive `Tensor` of the same `dtype` and shape as
      `mean_reversion` or a callable with the following properties: (a)  The
        callable should accept a scalar `Tensor` `t` and a 1-D `Tensor` `r(t)`
        of shape `[num_samples]` and returns a 2-D `Tensor` of shape
        `[num_samples, dim]`. The variable `t`  stands for time and `r(t)` is
        the short rate at time `t`.  The function returns instantaneous
        volatility `sigma(t) = sigma(t, r(t))`. When `volatility` is specified
        is a real `Tensor`, each factor is assumed to have a constant
        instantaneous volatility  and the  model is effectively a Gaussian HJM
        model. Corresponds to the instantaneous volatility of each factor.
    corr_matrix: A `Tensor` of shape `[dim, dim]` and the same `dtype` as
      `mean_reversion`. Corresponds to the correlation matrix `Rho`.
      Default value: None, meaning the factors are uncorrelated.
    notional: An optional `Tensor` of same dtype and compatible shape as
      `strikes`specifying the notional amount for the cap (or floor).
       Default value: None in which case the notional is set to 1.
    is_cap: A boolean `Tensor` of a shape compatible with `strikes`. Indicates
      whether the option is a Cap (if True) or a Floor (if False). If not
      supplied, Caps are assumed.
    num_samples: Positive scalar `int32` `Tensor`. The number of simulation
      paths during Monte-Carlo valuation.
      Default value: The default value is 1.
    random_type: Enum value of `RandomType`. The type of (quasi)-random number
      generator to use to generate the simulation paths.
      Default value: `None` which maps to the standard pseudo-random numbers.
    seed: Seed for the random number generator. The seed is only relevant if
      `random_type` is one of `[STATELESS, PSEUDO, HALTON_RANDOMIZED,
      PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC]`. For `PSEUDO`,
      `PSEUDO_ANTITHETIC` and `HALTON_RANDOMIZED` the seed should be an Python
      integer. For `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as
      an integer `Tensor` of shape `[2]`.
      Default value: `None` which means no seed is set.
    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or
      Halton sequence to skip. Used only when `random_type` is 'SOBOL',
      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.
      Default value: `0`.
    time_step: Scalar real `Tensor`. Maximal distance between time grid points
      in Euler scheme. Relevant when Euler scheme is used for simulation.
      Default value: `None`.
    dtype: The default dtype to use when converting values to `Tensor`s.
      Default value: `None` which means that default dtypes inferred by
        TensorFlow are used.
    name: Python string. The name to give to the ops created by this class.
      Default value: `None` which maps to the default name
        `hjm_cap_floor_price`.

  Returns:
    A `Tensor` of real dtype and shape  strikes.shape[:-1] containing
    the computed option prices. For caplets that have reset in the past
    (expiries<0), the function sets the corresponding caplet prices to 0.0.
  """
  name = name or 'hjm_cap_floor_price'
  with tf.name_scope(name):
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    dtype = dtype or strikes.dtype
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    maturities = tf.convert_to_tensor(maturities, dtype=dtype,
                                      name='maturities')
    daycount_fractions = tf.convert_to_tensor(daycount_fractions, dtype=dtype,
                                              name='daycount_fractions')
    notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
    is_cap = tf.convert_to_tensor(is_cap, dtype=tf.bool, name='is_cap')
    is_call_options = ~is_cap
    bond_option_strikes = 1.0 / (1.0 + daycount_fractions * strikes)

    # The dimension of `caplet_prices` is going to be strikes.shape
    caplet_prices = zcb.bond_option_price(
        strikes=bond_option_strikes,
        expiries=expiries,
        maturities=maturities,
        discount_rate_fn=reference_rate_fn,
        dim=dim,
        mean_reversion=mean_reversion,
        volatility=volatility,
        corr_matrix=corr_matrix,
        is_call_options=is_call_options,
        num_samples=num_samples,
        random_type=random_type,
        seed=seed,
        skip=skip,
        time_step=time_step,
        dtype=dtype,
        name=name + '_bond_option')

    caplet_prices = tf.where(
        expiries < 0.0, tf.zeros_like(expiries), caplet_prices)

    cap_prices = tf.math.reduce_sum(
        notional * (1.0 + daycount_fractions * strikes) * caplet_prices,
        axis=-1)
    return cap_prices
