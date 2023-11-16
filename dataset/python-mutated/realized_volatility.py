"""Implements realized volatility statistics."""
import enum
import tensorflow.compat.v2 as tf
from tf_quant_finance.math import diff_ops

@enum.unique
class PathScale(enum.Enum):
    """Represents what scale a path is on.

  * `ORIGINAL`: Represents a path on its original scale.
  * `LOG`: Represents log scale values of a path.
  """
    ORIGINAL = 1
    LOG = 2

@enum.unique
class ReturnsType(enum.Enum):
    """Represents types of return processes.

  * `ABS`: Represents absolute returns.
  * `LOG`: Represents log returns.
  """
    ABS = 1
    LOG = 2

def realized_volatility(sample_paths, times=None, scaling_factors=None, returns_type=ReturnsType.LOG, path_scale=PathScale.ORIGINAL, axis=-1, dtype=None, name=None):
    if False:
        print('Hello World!')
    "Calculates the total realized volatility for each path.\n\n  With `t_i, i=0,...,N` being a discrete sequence of times at which a series\n  `S_{t_k}, i=0,...,N` is observed. The logarithmic returns (`ReturnsType.LOG`)\n  process is given by:\n\n  ```\n  R_k = log(S_{t_{k}} / S_{t_{k-1}})^2\n  ```\n\n  Whereas for absolute returns (`ReturnsType.ABS`) it is given by:\n\n  ```\n  R_k = |S_{t_k}} - S_{t_{k-1}})| / |S_{t_{k-1}}|\n  ```\n\n  Letting `dt_k = t_k - t_{k-1}` the realized variance is then calculated as:\n\n  ```\n  V = c * f( \\Sum_{k=1}^{N-1} R_k / dt_k )\n  ```\n\n  Where `f` is the square root for logarithmic returns and the identity function\n  for absolute returns. If `times` is not supplied then it is assumed that\n  `dt_k = 1` everywhere. The arbitrary scaling factor `c` enables various\n  flavours of averaging or annualization (for examples of which see [1] or\n  section 9.7 of [2]).\n\n  #### Examples\n\n  Calculation of realized logarithmic volatility as in [1]:\n\n  ```python\n  import tensorflow as tf\n  import tf_quant_finance as tff\n  dtype=tf.float64\n  num_samples = 1000\n  num_times = 252\n  seed = (1, 2)\n  annual_vol = 20\n  sigma = annual_vol / (100 * np.sqrt(num_times - 1))\n  mu = -0.5*sigma**2\n\n  gbm = tff.models.GeometricBrownianMotion(mu=mu, sigma=sigma, dtype=dtype)\n  sample_paths = gbm.sample_paths(\n      times=range(num_times),\n      num_samples=num_samples,\n      seed=seed,\n      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)\n\n  annualization = 100 * np.sqrt( (num_times / (num_times - 1)) )\n  tf.math.reduce_mean(\n    realized_volatility(sample_paths,\n                        scaling_factors=annualization,\n                        path_scale=PathScale.ORIGINAL,\n                        axis=1))\n  # 20.03408344960287\n  ```\n\n  Carrying on with the same paths the realized absolute volatility (`RV_d2` in\n  [3]) is:\n\n  ```\n  scaling = 100 * np.sqrt((np.pi/(2 * (num_times-1))))\n  tf.math.reduce_mean(\n    realized_volatility(sample_paths,\n                        scaling_factors=scaling,\n                        returns_type=ReturnsType.ABS,\n                        path_scale=PathScale.LOG))\n  # 19.811590402553158\n  ```\n\n  #### References:\n  [1]: CBOE. Summary Product Specifications Chart for S&P 500 Variance Futures.\n  2012.\n  https://cdn.cboe.com/resources/futures/sp_500_variance_futures_contract.pdf\n  [2]: Iain J. Clark. Foreign exchange option pricing - A Practitioner's\n  guide. Chapter 5. 2011.\n  [3]: Zhu, S.P. and Lian, G.H., 2015. Analytically pricing volatility swaps\n  under stochastic volatility. Journal of Computational and Applied Mathematics.\n\n  Args:\n    sample_paths: A real `Tensor` of shape\n      `batch_shape_0 + [N] + batch_shape_1`.\n    times: A real `Tensor` of shape compatible with `batch_shape_0 + [N] +\n      batch_shape_1`. The times represented on the axis of interest (the `t_k`).\n      Default value: None. Resulting in the assumption of unit time increments.\n    scaling_factors: An optional real `Tensor` of shape compatible with\n      `batch_shape_0 + batch_shape_1`. Any scaling factors to be applied to the\n      result (e.g. for annualization).\n      Default value: `None`. Resulting in `c=1` in the above calculation.\n    returns_type: Value of ReturnsType. Indicates which definition of returns\n      should be used.\n      Default value: ReturnsType.LOG, representing logarithmic returns.\n    path_scale: Value of PathScale. Indicates which space the supplied\n      `sample_paths` are in. If required the paths will then be transformed onto\n      the appropriate scale.\n      Default value: PathScale.ORIGINAL.\n    axis: Python int. The axis along which to calculate the statistic.\n      Default value: -1 (the final axis).\n    dtype: `tf.DType`. If supplied the dtype for the input and output `Tensor`s.\n      Default value: `None` leading to use of `sample_paths`.\n    name: Python str. The name to give to the ops created by this function.\n      Default value: `None` which maps to 'realized_volatility'.\n\n  Returns:\n    Tensor of shape equal to `batch_shape_0 + batch_shape_1` (i.e. with axis\n      `axis` having been reduced over).\n  "
    with tf.name_scope(name or 'realized_volatility'):
        sample_paths = tf.convert_to_tensor(sample_paths, dtype=dtype, name='sample_paths')
        dtype = dtype or sample_paths.dtype
        if returns_type == ReturnsType.LOG:
            component_transform = lambda t: tf.pow(t, 2)
            result_transform = tf.math.sqrt
            if path_scale == PathScale.ORIGINAL:
                transformed_paths = tf.math.log(sample_paths)
            elif path_scale == PathScale.LOG:
                transformed_paths = sample_paths
        elif returns_type == ReturnsType.ABS:
            component_transform = tf.math.abs
            result_transform = tf.identity
            if path_scale == PathScale.ORIGINAL:
                transformed_paths = sample_paths
            elif path_scale == PathScale.LOG:
                transformed_paths = tf.math.exp(sample_paths)
        diffs = component_transform(diff_ops.diff(transformed_paths, order=1, exclusive=True, axis=axis))
        denominators = 1
        if times is not None:
            times = tf.convert_to_tensor(times, dtype=dtype, name='times')
            denominators = diff_ops.diff(times, order=1, exclusive=True, axis=axis)
        if returns_type == ReturnsType.ABS:
            slices = transformed_paths.shape.rank * [slice(None)]
            slices[axis] = slice(None, -1)
            denominators = denominators * component_transform(transformed_paths[slices])
        path_statistics = result_transform(tf.math.reduce_sum(diffs / denominators, axis=axis))
        if scaling_factors is not None:
            scaling_factors = tf.convert_to_tensor(scaling_factors, dtype=dtype, name='scaling_factors')
            return scaling_factors * path_statistics
        return path_statistics