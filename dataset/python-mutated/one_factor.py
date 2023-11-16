"""One factor Hull-White model with time-dependent parameters."""
from typing import Callable, Union
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.models.hull_white import vector_hull_white
__all__ = ['HullWhiteModel1F']

class HullWhiteModel1F(vector_hull_white.VectorHullWhiteModel):
    """One Factor Hull-White Model.

  Represents the Ito process:

  ```None
    dr(t) = (theta(t) - a(t) * r(t)) dt + sigma(t) * dW_{r}(t)
  ```
  where `W_{r}` is a 1D Brownian motion.
  `theta`, `a`, `sigma`, are positive functions of time.
  `a` correspond to the mean-reversion rate, `sigma` is the volatility of
  the process, `theta(t)` is the function that determines long run behaviour
  of the process `r(t)` and is defined to match the market data through the
  instantaneous forward rate matching:

  ```None
  \\theta = df(t) / dt + a * f(t) + 0.5 * sigma**2 / a
           * (1 - exp(-2 * a *t)), 1 <= i <= n
  ```
  where `f(t)` is the instantaneous forward rate at time `0` for a maturity
  `t` and `df(t)/dt` is the gradient of `f` with respect to the maturity.
  See Section 3.3.1 of [1] for details.

  If the parameters `a` and `sigma` are piecewise constant functions, the
  process is sampled exactly. Otherwise, Euler sampling is used.

  #### Example. Hull-White processes with piecewise constant coefficients.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64
  # Mean-reversion is a piecewise constant function.
  mean_reversion = tff.math.piecewise.PiecewiseConstantFunc(
      jump_locations=[1, 2, 3, 4],
      values=[0.1, 0.2, 0.3, 0.4, 0.5],
      dtype=dtype)
  # Volatility is a piecewise constant function.
  volatility = tff.math.piecewise.PiecewiseConstantFunc(
      jump_locations=[0.1, 2.],
      values=[0.1, 0.2, 0.1],
      dtype=dtype)
  initial_discount_rate_fn = lambda *args: [0.01]
  process = tff.models.hull_white.HullWhiteModel1F(
      mean_reversion=mean_reversion,
      volatility=volatility,
      initial_discount_rate_fn=initial_discount_rate_fn,
      dtype=dtype)
  # Sample 10000 paths using Sobol numbers as a random type.
  times = np.linspace(0., 1.0, 10)
  num_samples = 10000  # number of trajectories
  paths = process.sample_paths(
      times,
      num_samples=num_samples,
      times_grid=times,
      random_type=tff.math.random.RandomType.SOBOL)
  # Compute mean for each Hull-White process at the terminal value
  tf.math.reduce_mean(paths[:, -1, 0], axis=0)
  # Expected value: 0.02996

  ```

  #### References:
    [1]: D. Brigo, F. Mercurio. Interest Rate Models. 2007.
  """

    def __init__(self, mean_reversion: Union[types.RealTensor, Callable[..., types.RealTensor]], volatility: Union[types.RealTensor, Callable[..., types.RealTensor]], initial_discount_rate_fn: Callable[..., types.RealTensor], dtype: tf.DType=None, name: str=None):
        if False:
            print('Hello World!')
        'Initializes Hull-White Model.\n\n    Args:\n      mean_reversion: A real positive scalar `Tensor` or a Python callable. The\n        callable can be one of the following:\n          (a) A left-continuous piecewise constant object (e.g.,\n          `tff.math.piecewise.PiecewiseConstantFunc`) that has a property\n          `is_piecewise_constant` set to `True`. In this case the object\n          should have a method `jump_locations(self)` that returns a\n          `Tensor` of shape `[num_jumps]`. `mean_reversion(t)` should return a\n          `Tensor` `t.shape`,\n          where `t` is a rank 1 `Tensor` of the same `dtype` as the output.\n          See example in the class docstring.\n         (b) A callable that accepts scalars (stands for time `t`) and returns a\n         scalar `Tensor` of the same `dtype` as the input.\n        Corresponds to the mean reversion rate.\n      volatility: A real positive scalar `Tensor` of the same `dtype` as\n        `mean_reversion` or a callable with the same specs as above.\n        Corresponds to the lond run price variance.\n      initial_discount_rate_fn: A Python callable that accepts expiry time as a\n        real `Tensor` of the same `dtype` as `mean_reversion` and returns\n        a `Tensor` of either shape `input_shape`. Corresponds to the initial\n        discount rates at time `t=0` such that\n        `P(0,t) = exp(-y(t) * t)` where `P(0,t)` denotes the initial discount\n        bond prices.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: `None` which maps to `tf.float32`.\n      name: Python string. The name to give to the ops created by this class.\n        Default value: `None` which maps to the default name `hull_white_model`.\n    '
        name = name or 'hull_white_one_factor'
        super(HullWhiteModel1F, self).__init__(1, mean_reversion, volatility, initial_discount_rate_fn, corr_matrix=None, dtype=dtype, name=name)