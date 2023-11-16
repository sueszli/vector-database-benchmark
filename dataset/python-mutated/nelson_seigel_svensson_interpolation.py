"""Nelson Seigel Svensson interpolation method."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
__all__ = ['interpolate', 'SvenssonParameters']

@tff_utils.dataclass
class SvenssonParameters:
    """Nelson Seigel Svensson interpolation parameters.

  Attributes:
    beta_0: A real `Tensor` of arbitrary shape `batch_shape`.
    beta_1: A real `Tensor` of arbitrary shape `batch_shape`.
    beta_2: A real `Tensor` of arbitrary shape `batch_shape`.
    beta_3: A real `Tensor` of arbitrary shape `batch_shape`.
    tau_1: A real `Tensor` of arbitrary shape `batch_shape`.
    tau_2: A real `Tensor` of arbitrary shape `batch_shape`.
  """
    beta_0: types.RealTensor
    beta_1: types.RealTensor
    beta_2: types.RealTensor
    beta_3: types.RealTensor
    tau_1: types.RealTensor
    tau_2: types.RealTensor

def interpolate(interpolation_times: types.RealTensor, svensson_parameters: SvenssonParameters, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        for i in range(10):
            print('nop')
    "Performs Nelson Seigel Svensson interpolation for supplied points.\n\n  Given a set of interpolation times and the parameters for the nelson seigel\n  svensson model, this function returns the interpolated values for the yield\n  curve. We assume that the parameters are already computed using a fitting\n  technique.\n  ```None\n  r(T) = beta_0 +\n         beta_1 * (1-exp(-T/tau_1))/(T/tau_1) +\n         beta_2 * ((1-exp(-T/tau_1))/(T/tau_1) - exp(-T/tau_1)) +\n         beta_3 * ((1-exp(-T/tau_2))/(T/tau_2) - exp_(-T/tau_2))\n  ```\n\n  Where `T` represents interpolation times and\n  `beta_i`'s and `tau_i`'s are paramters for the model.\n\n  #### Example\n  ```python\n  import tf_quant_finance as tff\n  interpolation_times = [5., 10., 15., 20.]\n  svensson_parameters =\n  tff.rates.nelson_svensson.interpolate.SvenssonParameters(\n        beta_0=0.05, beta_1=-0.01, beta_2=0.3, beta_3=0.02,\n        tau_1=1.5, tau_2=20.0)\n  result = interpolate(interpolation_times, svensson_parameters)\n  # Expected_result\n  # [0.12531, 0.09667, 0.08361, 0.07703]\n  ```\n\n  #### References:\n    [1]: Robert Müller. A technical note on the Svensson model as applied to\n    the Swiss term structure.\n    BIS Papers No 25, Mar 2015.\n    https://www.bis.org/publ/bppdf/bispap25l.pdf\n\n  Args:\n    interpolation_times: The times at which interpolation is desired. A N-D\n      `Tensor` of real dtype where the first N-1 dimensions represent the\n      batching dimensions.\n    svensson_parameters: An instance of `SvenssonParameters`. All parameters\n      within should be real tensors.\n    dtype: Optional tf.dtype for `interpolation_times`. If not specified, the\n      dtype of the inputs will be used.\n    name: Python str. The name prefixed to the ops created by this function. If\n      not supplied, the default name 'nelson_svensson_interpolation' is used.\n\n  Returns:\n    A N-D `Tensor` of real dtype with the same shape as `interpolations_times`\n      containing the interpolated yields.\n  "
    name = name or 'nelson_svensson_interpolation'
    with tf.compat.v1.name_scope(name):
        interpolation_times = tf.convert_to_tensor(interpolation_times, dtype=dtype)
        dtype = dtype or interpolation_times.dtype
        yield_part0 = svensson_parameters.beta_0
        yield_part1 = svensson_parameters.beta_1 * _integrated_exp_term(interpolation_times, svensson_parameters.tau_1)
        yield_part2 = svensson_parameters.beta_2 * (_integrated_exp_term(interpolation_times, svensson_parameters.tau_1) - _exp_term(interpolation_times, svensson_parameters.tau_1))
        yield_part3 = svensson_parameters.beta_3 * (_integrated_exp_term(interpolation_times, svensson_parameters.tau_2) - _exp_term(interpolation_times, svensson_parameters.tau_2))
        interpolated_yields = yield_part0 + yield_part1 + yield_part2 + yield_part3
    return interpolated_yields

def _exp_term(x, y):
    if False:
        print('Hello World!')
    return tf.math.exp(-tf.math.divide_no_nan(x, y))

def _integrated_exp_term(x, y):
    if False:
        for i in range(10):
            print('nop')
    return (1 - _exp_term(x, y)) / tf.math.divide_no_nan(x, y)