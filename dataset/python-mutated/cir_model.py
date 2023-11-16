"""Cox–Ingersoll–Ross model."""
from typing import Optional
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.math import random_ops as random
from tf_quant_finance.models import generic_ito_process

class CirModel(generic_ito_process.GenericItoProcess):
    """Cox–Ingersoll–Ross model.

  Represents the Ito process:

  ```None
    dX_i(t) = (a - k*X_i(t)) * dt +  sigma * sqrt(X_i(t)) * dW_i(t)
  ```
  where
    a / k: Corresponds to the long term mean.
    k: Corresponds to the speed of reversion.
    sigma: Corresponds to the instantaneous volatility.

  See [1] for details.

  #### References:
    [1]: A. Alfonsi. Affine Diffusions and Related Processes: Simulation,
      Theory and Applications
  """

    def __init__(self, theta: types.RealTensor, mean_reversion: types.RealTensor, sigma: types.RealTensor, dtype: Optional[tf.DType]=None, name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the CIR Model.\n\n    Args:\n      theta: A positive scalar `Tensor` with shape `batch_shape` + [1].\n      mean_reversion: A positive scalar `Tensor` of the same dtype and shape as\n        `theta`. Means speed of reversion.\n      sigma: A scalar `Tensor` of the same dtype and shape as `theta`.Means\n        volatility.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: `None` which maps to `tf.float32`.\n      name: Python string. The name to give to the ops created by this class.\n        Default value: `None` which maps to the default name `cir_model`.\n    '
        dim = 1
        dtype = dtype or tf.float32
        name = name or 'cir_model'
        with tf.name_scope(name):

            def _convert_param_to_tensor(param):
                if False:
                    return 10
                'Converts `param` to `Tesnor`.\n\n        Args:\n          param: `Scalar` or `Tensor` with shape `batch_shape` + [1].\n\n        Returns:\n          `param` if it `Tensor`, if it is `Scalar` convert it to `Tensor` with\n          [1] shape.\n        '
                param_t = tf.convert_to_tensor(param, dtype=dtype)
                return param_t * tf.ones(shape=dim, dtype=dtype)

            def _get_batch_shape(param):
                if False:
                    i = 10
                    return i + 15
                '`param` must has shape `batch_shape + [1]`.'
                param_shape = tff_utils.get_shape(param)
                return param_shape[:-1]
            self._theta = _convert_param_to_tensor(theta)
            self._mean_reversion = _convert_param_to_tensor(mean_reversion)
            self._sigma = _convert_param_to_tensor(sigma)
            self._batch_shape = _get_batch_shape(self._theta)
            self._batch_shape_rank = len(self._batch_shape)

            def _drift_fn(t, x):
                if False:
                    while True:
                        i = 10
                del t
                expand_rank = tff_utils.get_shape(x).rank - self._batch_shape_rank - 1
                theta_expand = self._expand_param_on_rank(self._theta, expand_rank, axis=-2)
                mean_reversion_expand = self._expand_param_on_rank(self._mean_reversion, expand_rank, axis=-2)
                return theta_expand - mean_reversion_expand * x

            def _volatility_fn(t, x):
                if False:
                    return 10
                del t
                expand_rank = len(tff_utils.get_shape(x)) - self._batch_shape_rank - 1
                sigma_expand = self._expand_param_on_rank(self._sigma, expand_rank, axis=-2)
                return tf.expand_dims(sigma_expand * tf.sqrt(x), axis=-1)
        super(CirModel, self).__init__(dim, _drift_fn, _volatility_fn, dtype, name)

    def sample_paths(self, times: types.RealTensor, initial_state: Optional[types.RealTensor]=None, num_samples: int=1, random_type: Optional[random.RandomType]=None, seed: Optional[int]=None, name: Optional[str]=None) -> types.RealTensor:
        if False:
            i = 10
            return i + 15
        'Returns a sample of paths from the process.\n\n    Using exact simulation method from [1].\n\n    Args:\n      times: Rank 1 `Tensor` of positive real values. The times at which the\n        path points are to be evaluated.\n      initial_state: A `Tensor` of the same `dtype` as `times` and of shape\n        broadcastable with `batch_shape + [num_samples, 1]`. Represents the\n        initial state of the Ito process. `batch_shape` is the shape of the\n        independent stochastic processes being modelled and is inferred from the\n        initial state `x0`.\n        Default value: `None` which maps to a initial state of ones.\n      num_samples: Positive scalar `int`. The number of paths to draw.\n      random_type: `STATELESS` or `PSEUDO` type from `RandomType` Enum. The type\n        of (quasi)-random number generator to use to generate the paths.\n      seed: The seed for the random number generator.\n        For `PSEUDO` random type: it is an Integer.\n        For `STATELESS` random type: it is an integer `Tensor` of shape `[2]`.\n          In this case the algorithm samples random numbers with seeds `[seed[0]\n          + i, seed[1] + j], i in {0, 1}, j in {0, 1, ..., num_times}`, where\n          `num_times` is the size of `times`.\n        Default value: `None` which means no seed is set, but it works only with\n          `PSEUDO` random type. For `STATELESS` it has to be provided.\n      name: Str. The name to give this op.\n        Default value: `sample_paths`.\n\n    Returns:\n      A `Tensor`s of shape batch_shape + [num_samples, num_times, 1] where\n      `num_times` is\n      the size of the `times`.\n\n    Raises:\n      ValueError: If `random_type` or `seed` is not supported.\n\n    ## Example\n\n    ```python\n    import tensorflow as tf\n    import tf_quant_finance as tff\n\n    # In this example `batch_shape` is 2, so parameters has shape [2, 1]\n    process = tff.models.CirModel(\n        theta=[[0.02], [0.03]],\n        mean_reversion=[[0.5], [0.4]],\n        sigma=[[0.1], [0.5]],\n        dtype=tf.float64)\n\n    num_samples = 5\n    # `initial_state` has shape [num_samples, 1]\n    initial_state=[[0.1], [0.2], [0.3], [0.4], [0.5]]\n    times = [0.1, 0.2, 1.0]\n    samples = process.sample_paths(\n        times=times,\n        num_samples=num_samples,\n        initial_state=initial_state)\n    # `samples` has shape [2, 5, 3, 1]\n    ```\n\n    #### References:\n    [1]: A. Alfonsi. Affine Diffusions and Related Processes: Simulation,\n      Theory and Applications\n    '
        name = name or self._name + '_sample_path'
        with tf.name_scope(name):
            element_shape = self._batch_shape + [num_samples, self._dim]
            theta = self._expand_param_on_rank(self._theta, 1, axis=-2)
            mean_reversion = self._expand_param_on_rank(self._mean_reversion, 1, axis=-2)
            sigma = self._expand_param_on_rank(self._sigma, 1, axis=-2)
            if initial_state is None:
                initial_state = tf.ones(element_shape, dtype=self._dtype, name='initial_state')
            else:
                initial_state = tf.convert_to_tensor(initial_state, dtype=self._dtype, name='initial_state') + tf.zeros(element_shape, dtype=self._dtype)
            times = tf.convert_to_tensor(times, dtype=self._dtype, name='times')
            num_requested_times = tff_utils.get_shape(times)[0]
            if random_type is None:
                random_type = random.RandomType.PSEUDO
            if random_type == random.RandomType.STATELESS and seed is None:
                raise ValueError('`seed` equal to None is not supported with STATELESS random type.')
            return self._sample_paths(theta=theta, mean_reversion=mean_reversion, sigma=sigma, element_shape=element_shape, times=times, num_requested_times=num_requested_times, initial_state=initial_state, num_samples=num_samples, random_type=random_type, seed=seed)

    def _sample_paths(self, theta, mean_reversion, sigma, element_shape, times, num_requested_times, initial_state, num_samples, random_type, seed):
        if False:
            print('Hello World!')
        'Returns a sample of paths from the process.'
        times = tf.concat([[0], times], -1)
        dts = tf.expand_dims(tf.expand_dims(times[1:] - times[:-1], axis=-1), axis=-1)
        (poisson_fn, gamma_fn, poisson_seed_fn, gamma_seed_fn) = self._get_distributions(random_type)

        def _sample_at_time(i, update_idx, current_x, samples):
            if False:
                print('Hello World!')
            dt = dts[i]
            zeta = tf.where(tf.math.equal(mean_reversion, tf.zeros_like(mean_reversion)), dt, (1 - tf.math.exp(-mean_reversion * dt)) / mean_reversion)
            c = tf.math.divide_no_nan(tf.constant(4, dtype=self._dtype), sigma ** 2 * zeta)
            d = c * tf.math.exp(-mean_reversion * dt)
            poisson_rv = poisson_fn(shape=element_shape, lam=d * current_x / 2, seed=poisson_seed_fn(seed, i), dtype=self._dtype)
            gamma_param_alpha = poisson_rv + 2 * theta / sigma ** 2
            gamma_param_beta = c / 2
            new_x = gamma_fn(shape=element_shape, alpha=gamma_param_alpha, beta=gamma_param_beta, seed=gamma_seed_fn(seed, i), dtype=self._dtype)
            new_x = tf.where(c > 0, new_x, current_x)
            samples = samples.write(i, new_x)
            return (i + 1, update_idx, new_x, samples)
        cond_fn = lambda i, *args: i < num_requested_times
        samples = tf.TensorArray(dtype=self._dtype, size=num_requested_times, element_shape=element_shape, clear_after_read=False)
        (_, _, _, samples) = tf.while_loop(cond_fn, _sample_at_time, (0, 0, initial_state, samples), maximum_iterations=num_requested_times)
        samples = samples.stack()
        samples_rank = len(tff_utils.get_shape(samples))
        perm = [batch_idx for batch_idx in range(1, samples_rank - 2)] + [samples_rank - 2, 0, samples_rank - 1]
        return tf.transpose(samples, perm=perm)

    def _expand_param_on_rank(self, param, expand_rank, axis):
        if False:
            for i in range(10):
                print('nop')
        'Adds dimensions to `param`, not inplace.\n\n    Args:\n      param: initial element.\n      expand_rank: is amount of dimensions that need to be added.\n      axis: is axis where to place these dimensions.\n\n    Returns:\n      New `Tensor`.\n    '
        param_tensor = tf.convert_to_tensor(param, dtype=self._dtype)
        param_expand = param_tensor
        for _ in range(expand_rank):
            param_expand = tf.expand_dims(param_expand, axis)
        return param_expand

    @staticmethod
    def _get_distributions(random_type):
        if False:
            print('Hello World!')
        'Returns the distribution depending on the `random_type`.\n\n    Args:\n      random_type: `STATELESS` or `PSEUDO` type from `RandomType` Enum.\n\n    Returns:\n     Tuple (Poisson distribution, Gamma distribution, function to generate\n     seed for Poisson distribution, function to generate seed for Gamma\n     distribution).\n    '
        if random_type == random.RandomType.STATELESS:
            poisson_seed_fn = lambda seed, i: tf.stack([seed[0], seed[1] + i])
            gamma_seed_fn = lambda seed, i: tf.stack([seed[0] + 1, seed[1] + i])
            return (tf.random.stateless_poisson, tf.random.stateless_gamma, poisson_seed_fn, gamma_seed_fn)
        elif random_type == random.RandomType.PSEUDO:
            seed_fn = lambda seed, _: seed
            return (tf.random.poisson, tf.random.gamma, seed_fn, seed_fn)
        else:
            raise ValueError('Only STATELESS and PSEUDO random types are supported.')