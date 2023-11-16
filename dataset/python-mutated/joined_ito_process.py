"""Join a sequence of Ito Processes with specified correlations."""
import tensorflow.compat.v2 as tf
from tf_quant_finance.models import euler_sampling
from tf_quant_finance.models import generic_ito_process
from tf_quant_finance.models import ito_process
from tf_quant_finance.models import utils

class JoinedItoProcess(generic_ito_process.GenericItoProcess):
    """Join of Ito Processes with specified time dependent correlations.

  For a sequence of Ito processes `I_1, .., I_n` of dimensions `d_1,.., d_n`,
  the class initializes a process `I` of dimension `d_1 + .. + d_n` with
  marginal proceses `I_i` and a correlation function `Corr(t)`. That is, let the
  Ito Process `I_i` describe an SDE

  ```None
  dX^i = a_i(t, X^i_t) dt + b_i(t, X^i_t) dW^i_t
  ```

  where `a_i(t, x)` is a function taking values in `R^{d_i}`, `b_i(t, X_t)` is a
  function taking values in `d_i x d_i` matrices, `W_i` is a `d_i`-dimensional
  Brownian motion.

  Then `I` describes an SDE for the joint process `(X_1,..., X_n)` of dimension
  `d:= d_1 + ... + d_n`

  ```None
  dX^i = a_i(t, X^i_t) dt + b_i(t, X^i_t) dB^i_t,
  ```

  where `(B_1, ..., B_n) = chol(t) * (W_1, ..., W_n)` for a Cholesky
  decomposition `chol(t)` of the correlation matrix `Corr(t)` at time `t`.
  Here `(W_1, ..., W_n)` is `d`-dimensional vector and `Corr(t)` is a `d x d`
  correlation matrix.

  `Corr(t)` is represented as a block-diagonal formed of a list of matrices
  `[m_1(t), m_2(t), ..., m_k(t)]` with `sum(rank(m_i)) = d`.

  # Example. # Black-scholes and Heston model join.
  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff

  dtype = tf.float64
  # Define Black scholes model with zero rate and volatility `0.1`
  sigma = 0.1
  def drift_fn(t , x):
    return -sigma**2 / 2
  def vol_fn(t , x):
    return sigma * tf.ones([1, 1], dtype=x.dtype)
  black_scholes_process = tff.models.GenericItoProcess(
      dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=dtype)
  # Define Heston Model
  epsilon = tff.math.piecewise.PiecewiseConstantFunc(
      jump_locations=[0.5], values=[0.01, 0.02], dtype=np.float64)
  heston_process = tff.models.heston_model.HestonModel(
      kappa=1.0, theta=0.04, epsilon=epsilon, rho=0.2, dtype=dtype)
  # Define join process where correlation between `black_scholes_process` and
  # log-asset price of the `heston_process` is 0.5.
  corr_structure = [[[1.0, 0.5], [0.5, 1.0]], [1.0]]
  # `corr_structure` corresponds to a `3 x 3` correlation matrix. Here Brownian
  # motion of `black_scholes_process` is correlated only with the 1st dimension
  # of `heston_process` but not with the second one.
  join_process = JoinedItoProcess(
      processes=[black_scholes_process, heston_process],
      corr_structure=corr_structure)
  # Sample 100,000 sample paths at times [0.1, 1.0] from the join process using
  # Sobol random sequence
  times = [0.1, 1.0]
  # Wrap sample_paths method with a tf.function
  sample_paths_fn = tf.function(process.sample_paths)
  samples = sample_paths_fn(
      times=times, time_step=0.01, num_samples=100000,
      initial_state=np.array([0.0, 1.0, 0.04]),
      random_type=random.RandomType.SOBOL)
  # Estimated correlations.
  np.corrcoef(samples[:, -1, :], rowvar=False)
  # Expected result:
  # [[1.        , 0.49567078, 0.08128067],
  #  [0.49567078, 1.        , 0.16580689],
  #  [0.08128067, 0.16580689, 1.        ]]
  ```
  """

    def __init__(self, processes, corr_structure, dtype=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes a JoinedItoProcess.\n\n    Takes a list of `processes` which are instances of `tff.models.ItoProcess`\n    and a list `corr_structure` of correlation matrices and creates an Ito\n    process that joins `processes` using the correlation structure.\n    `corr_structure` describes block-diagonal structure of correlations for\n    the Brownian motions in `processes`. For example, if the dimension of the\n    JoinedItoProcess is `3` and\n    `corr_structure = [[[1.0, 0.5], [0.5, 1.0]], [1.0]]`, then the introduced\n    correlation is\n    `Corr(t) = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]`,\n    where `Corr(t)` is the same as in the `JoinedItoProcess` docstring.\n\n    Args:\n      processes: A sequence of instances of `tff.models.ItoProcess`. All\n        processes should have the same `dtype.`\n      corr_structure: A list of correlation matrices. Each correlation matrix\n        is either a `Tensor` of the same `dtype` as the `processes` and\n        square shape  (i.e., `[d_i, d_i]` for some `d_i`) or a callable. The\n        callables should accept a scalar (stands for time `t`) and return a\n        square `Tensor`. The total dimension\n        `sum([m.shape[-1] for m in corr_structure]` of correlation\n        structure should be the same as the dimension of the `JoinedItoProcess`\n        `sum([p.dim() for p in processes])`.\n      dtype: The default `dtype` of the `processes`.\n        Default value: None which means that default dtypes inferred by\n          TensorFlow are used.\n      name: Python string. The name scope under which ops created by the methods\n        of this class are nested.\n        Default value: `None` which maps to the default name\n          `join_ito_process`.\n\n    Raises:\n      ValueError:\n        (a) If any of the `processes` is not an `ItoProcess`.\n        (b) If `processes` do not have the same `dtype`.\n    '
        self._name = name or 'join_ito_process'
        with tf.name_scope(self._name):
            self._processes = []
            dim = 0
            for process in processes:
                if not isinstance(process, ito_process.ItoProcess):
                    raise ValueError('All input process of JoinedItoProcess must be instances of the ItoProcess class.')
                self._processes.append(process)
                d = process.dim()
                dim += d
                if dtype is None:
                    dtype = process.dtype()
                elif dtype != process.dtype():
                    raise ValueError('All processes should have the same `dtype`')
            self._corr_structure = [corr if callable(corr) else tf.convert_to_tensor(corr, dtype=dtype, name='corr') for corr in corr_structure]
            self._dim = dim

            def _drift_fn(t, x):
                if False:
                    print('Hello World!')
                'Drift function of the JoinedItoProcess.'
                drifts = []
                i1 = 0
                i2 = 0
                for p in self._processes:
                    dim = p.dim()
                    i2 += dim
                    position = x[..., i1:i2]
                    drift = tf.convert_to_tensor(p.drift_fn()(t, position), dtype=dtype, name='drift')
                    drift = tf.broadcast_to(drift, position.shape)
                    drifts.append(drift)
                    i1 += dim
                return tf.concat(drifts, -1)

            def _vol_fn(t, x):
                if False:
                    print('Hello World!')
                'Volatility function of the JoinedItoProcess.'
                vols = []
                i1 = 0
                i2 = 0
                for p in self._processes:
                    dim = p.dim()
                    i2 += dim
                    position = x[..., i1:i2]
                    vol = tf.convert_to_tensor(p.volatility_fn()(t, position), dtype=dtype, name='volatility')
                    vol = tf.broadcast_to(vol, position.shape + [dim])
                    vols.append(vol)
                    i1 += dim
                vol = utils.block_diagonal_to_dense(*vols)
                corr_structure = _get_parameters(tf.expand_dims(t, -1), *self._corr_structure)
                cholesky_decomp = [tf.linalg.cholesky(m) for m in corr_structure]
                cholesky_decomp = utils.block_diagonal_to_dense(*cholesky_decomp)
                return tf.linalg.matmul(vol, cholesky_decomp)
            super().__init__(dim, _drift_fn, _vol_fn, dtype, name)

    def sample_paths(self, times, num_samples=1, initial_state=None, random_type=None, seed=None, time_step=None, swap_memory=True, skip=0, name=None):
        if False:
            print('Hello World!')
        "Returns a sample of paths from the process using Euler sampling.\n\n    Args:\n      times: Rank 1 `Tensor` of increasing positive real values. The times at\n        which the path points are to be evaluated.\n      num_samples: Positive scalar `int`. The number of paths to draw.\n        Default value: 1.\n      initial_state: `Tensor` of shape `[self._dim]`. The initial state of the\n        process.\n        Default value: None which maps to a zero initial state.\n      random_type: Enum value of `RandomType`. The type of (quasi)-random number\n        generator to use to generate the paths.\n        Default value: None which maps to the standard pseudo-random numbers.\n      seed: Seed for the random number generator. The seed is\n        only relevant if `random_type` is one of\n        `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,\n          STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and\n        `HALTON_RANDOMIZED` the seed should be an integer scalar `Tensor`. For\n        `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer\n        `Tensor` of shape `[2]`.\n        Default value: `None` which means no seed is set.\n      time_step: Real scalar `Tensor`. The maximal distance between time points\n        in grid in Euler scheme.\n      swap_memory: A Python bool. Whether GPU-CPU memory swap is enabled for\n        this op. See an equivalent flag in `tf.while_loop` documentation for\n        more details. Useful when computing a gradient of the op since\n        `tf.while_loop` is used to propagate stochastic process in time.\n        Default value: True.\n      skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n        Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n        'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.\n        Default value: `0`.\n      name: Python string. The name to give this op.\n        Default value: `None` which maps to `sample_paths` is used.\n\n    Returns:\n     A real `Tensor` of shape `[num_samples, k, n]` where `k` is the size of the\n     `times`, and `n` is the dimension of the process.\n\n    Raises:\n      ValueError: If `time_step` is not supplied.\n    "
        if time_step is None:
            raise ValueError('`time_step` has to be supplied for JoinedItoProcess `sample_paths` method.')
        name = name or self._name + 'sample_paths'
        with tf.name_scope(name):
            if initial_state is None:
                initial_state = tf.zeros(self._dim, dtype=self.dtype(), name='initial_state')
            elif isinstance(initial_state, (tuple, list)):
                initial_state = [tf.convert_to_tensor(state, dtype=self.dtype(), name='initial_state') for state in initial_state]
                initial_state = tf.stack(initial_state)
            else:
                initial_state = tf.convert_to_tensor(initial_state, dtype=self.dtype(), name='initial_state')
            samples = euler_sampling.sample(self.dim(), drift_fn=self.drift_fn(), volatility_fn=self.volatility_fn(), times=times, time_step=time_step, num_samples=num_samples, initial_state=initial_state, random_type=random_type, seed=seed, swap_memory=swap_memory, skip=skip, dtype=self.dtype())
            return samples

def _get_parameters(times, *params):
    if False:
        for i in range(10):
            print('nop')
    'Gets parameter values at at specified `times`.'
    res = []
    for param in params:
        if callable(param):
            t = tf.squeeze(times)
            param_value = tf.convert_to_tensor(param(t), dtype=times.dtype, name='param_value')
            res.append(tf.expand_dims(param_value, 0))
        else:
            res.append(param + tf.zeros(times.shape + param.shape, dtype=times.dtype))
    return res