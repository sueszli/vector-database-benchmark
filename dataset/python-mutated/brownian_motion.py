"""N-dimensional Brownian Motion.

Implements the Ito process defined by:

```
  dX_i = a_i(t) dt + Sum[S_{ij}(t) dW_{j}, 1 <= j <= n] for each i in {1,..,n}
```

where `dW_{j}, 1 <= j <= n` are n independent 1D Brownian increments. The
coefficient `a_i` is the drift and the matrix `S_{ij}` is the volatility of the
process.

For more details, see Ref [1].

#### References:
  [1]: Brent Oksendal. Stochastic Differential Equations: An Introduction with
    Applications. Springer. 2010.
"""
import tensorflow.compat.v2 as tf
from tf_quant_finance.math.random_ops import multivariate_normal as mvn
from tf_quant_finance.models.legacy import brownian_motion_utils as bmu
from tf_quant_finance.models.legacy import ito_process

class BrownianMotion(ito_process.ItoProcess):
    """The multi dimensional Brownian Motion."""

    def __init__(self, dim=1, drift=None, volatility=None, total_drift_fn=None, total_covariance_fn=None, dtype=None, name=None):
        if False:
            return 10
        'Initializes the Brownian motion class.\n\n    Represents the Ito process:\n\n    ```None\n      dX_i = a_i(t) dt + Sum(S_{ij}(t) dW_j for j in [1 ... n]), 1 <= i <= n\n\n    ```\n\n    `a_i(t)` is the drift rate of this process and the `S_{ij}(t)` is the\n    volatility matrix. Associated to these parameters are the integrated\n    drift and covariance functions. These are defined as:\n\n    ```None\n      total_drift_{i}(t1, t2) = Integrate(a_{i}(t), t1 <= t <= t2)\n      total_covariance_{ij}(t1, t2) = Integrate(inst_covariance_{ij}(t),\n                                                     t1 <= t <= t2)\n      inst_covariance_{ij}(t) = (S.S^T)_{ij}(t)\n    ```\n\n    Sampling from the Brownian motion process with time dependent parameters\n    can be done efficiently if the total drift and total covariance functions\n    are supplied. If the parameters are constant, the total parameters can be\n    automatically inferred and it is not worth supplying then explicitly.\n\n    Currently, it is not possible to infer the total drift and covariance from\n    the instantaneous values if the latter are functions of time. In this case,\n    we use a generic sampling method (Euler-Maruyama) which may be\n    inefficient. It is advisable to supply the total covariance and total drift\n    in the time dependent case where possible.\n\n    #### Example\n    The following is an example of a 1 dimensional brownian motion using default\n    arguments of zero drift and unit volatility.\n\n    ```python\n    process = bm.BrownianMotion()\n    times = np.array([0.2, 0.33, 0.7, 0.9, 1.88])\n    num_samples = 10000\n    with tf.Session() as sess:\n      paths = sess.run(process.sample_paths(\n          times,\n          num_samples=num_samples,\n          initial_state=np.array(0.1),\n          seed=1234))\n\n    # Compute the means at the specified times.\n    means = np.mean(paths, axis=0)\n    print (means)  # Mean values will be near 0.1 for each time\n\n    # Compute the covariances at the given times\n    covars = np.cov(paths.reshape([num_samples, 5]), rowvar=False)\n\n    # covars is a 5 x 5 covariance matrix.\n    # Expected result is that Covar(X(t), X(t\')) = min(t, t\')\n    expected = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))\n    print ("Computed Covars: {}, True Covars: {}".format(covars, expected))\n    ```\n\n    Args:\n      dim: Python int greater than or equal to 1. The dimension of the Brownian\n        motion.\n        Default value: 1 (i.e. a one dimensional brownian process).\n      drift: The drift of the process. The type and shape of the value must be\n        one of the following (in increasing order of generality) (a) A real\n        scalar `Tensor`. This corresponds to a time and component independent\n        drift. Every component of the Brownian motion has the same drift rate\n        equal to this value. (b) A real `Tensor` of shape `[dim]`. This\n        corresponds to a time independent drift with the `i`th component as the\n        drift rate of the `i`th component of the Brownian motion. (c) A Python\n        callable accepting a single positive `Tensor` of general shape (referred\n        to as `times_shape`) and returning a `Tensor` of shape `times_shape +\n        [dim]`. The input argument is the times at which the drift needs to be\n        evaluated. This case corresponds to a general time and direction\n        dependent drift rate.\n        Default value: None which maps to zero drift.\n      volatility: The volatility of the process. The type and shape of the\n        supplied value must be one of the following (in increasing order of\n        generality) (a) A positive real scalar `Tensor`. This corresponds to a\n        time independent, diagonal volatility matrix. The `(i, j)` component of\n        the full volatility matrix is equal to zero if `i != j` and equal to the\n        supplied value otherwise. (b) A positive real `Tensor` of shape `[dim]`.\n        This corresponds to a time independent volatility matrix with zero\n        correlation. The `(i, j)` component of the full volatility matrix is\n        equal to zero `i != j` and equal to the `i`th component of the supplied\n        value otherwise. (c) A positive definite real `Tensor` of shape `[dim,\n        dim]`. The full time independent volatility matrix. (d) A Python\n        callable accepting a single positive `Tensor` of general shape (referred\n        to as `times_shape`) and returning a `Tensor` of shape `times_shape +\n        [dim, dim]`. The input argument are the times at which the volatility\n        needs to be evaluated. This case corresponds to a general time and axis\n        dependent volatility matrix.\n        Default value: None which maps to a volatility matrix equal to identity.\n      total_drift_fn: Optional Python callable to compute the integrated drift\n        rate between two times. The callable should accept two real `Tensor`\n        arguments. The first argument contains the start times and the second,\n        the end times of the time intervals for which the total drift is to be\n        computed. Both the `Tensor` arguments are of the same dtype and shape.\n        The return value of the callable should be a real `Tensor` of the same\n        dtype as the input arguments and of shape `times_shape + [dim]` where\n        `times_shape` is the shape of the times `Tensor`. Note that it is an\n        error to supply this parameter if the `drift` is not supplied.\n        Default value: None.\n      total_covariance_fn: A Python callable returning the integrated covariance\n        rate between two times. The callable should accept two real `Tensor`\n        arguments. The first argument is the start times and the second is the\n        end times of the time intervals for which the total covariance is\n        needed. Both the `Tensor` arguments are of the same dtype and shape. The\n        return value of the callable is a real `Tensor` of the same dtype as the\n        input arguments and of shape `times_shape + [dim, dim]` where\n        `times_shape` is the shape of the times `Tensor`. Note that it is an\n        error to supply this argument if the `volatility` is not supplied.\n        Default value: None.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: None which means that default dtypes inferred by\n          TensorFlow are used.\n      name: str. The name scope under which ops created by the methods of this\n        class are nested.\n        Default value: None which maps to the default name `brownian_motion`.\n\n    Raises:\n      ValueError if the dimension is less than 1 or if total drift is supplied\n        but drift is not supplied or if the total covariance is supplied but\n        but volatility is not supplied.\n    '
        super(BrownianMotion, self).__init__()
        if dim < 1:
            raise ValueError('Dimension must be 1 or greater.')
        if drift is None and total_drift_fn is not None:
            raise ValueError('total_drift_fn must not be supplied if drift is not supplied.')
        if volatility is None and total_covariance_fn is not None:
            raise ValueError('total_covariance_fn must not be supplied if drift is not supplied.')
        self._dim = dim
        self._dtype = dtype
        self._name = name or 'brownian_motion'
        (drift_fn, total_drift_fn) = bmu.construct_drift_data(drift, total_drift_fn, dim, dtype)
        self._drift_fn = drift_fn
        self._total_drift_fn = total_drift_fn
        (vol_fn, total_covar_fn) = bmu.construct_vol_data(volatility, total_covariance_fn, dim, dtype)
        self._volatility_fn = vol_fn
        self._total_covariance_fn = total_covar_fn

    def dim(self):
        if False:
            i = 10
            return i + 15
        'The dimension of the process.'
        return self._dim

    def dtype(self):
        if False:
            return 10
        'The data type of process realizations.'
        return self._dtype

    def name(self):
        if False:
            i = 10
            return i + 15
        'The name to give to the ops created by this class.'
        return self._name

    def drift_fn(self):
        if False:
            i = 10
            return i + 15
        return lambda t, x: self._drift_fn(t)

    def volatility_fn(self):
        if False:
            return 10
        return lambda t, x: self._volatility_fn(t)

    def total_drift_fn(self):
        if False:
            return 10
        'The integrated drift of the process.\n\n    Returns:\n      None or a Python callable. None is returned if the input drift was a\n      callable and no total drift function was supplied.\n      The callable returns the integrated drift rate between two times.\n      It accepts two real `Tensor` arguments. The first argument is the\n      left end point and the second is the right end point of the time interval\n      for which the total drift is needed. Both the `Tensor` arguments are of\n      the same dtype and shape. The return value of the callable is\n      a real `Tensor` of the same dtype as the input arguments and of shape\n      `times_shape + [dim]` where `times_shape` is the shape of the times\n      `Tensor`.\n    '
        return self._total_drift_fn

    def total_covariance_fn(self):
        if False:
            for i in range(10):
                print('nop')
        'The total covariance of the process between two times.\n\n    Returns:\n      A Python callable returning the integrated covariances between two times.\n      The callable accepts two real `Tensor` arguments. The first argument\n      is the left end point and the second is the right end point of the time\n      interval for which the total covariance is needed.\n\n      The shape of the two input arguments and their dtypes must match.\n      The output of the callable is a `Tensor` of shape\n      `times_shape + [dim, dim]` containing the integrated covariance matrix\n      between the start times and end times.\n    '
        return self._total_covariance_fn

    def sample_paths(self, times, num_samples=1, initial_state=None, random_type=None, seed=None, swap_memory=True, name=None, **kwargs):
        if False:
            while True:
                i = 10
        'Returns a sample of paths from the process.\n\n    Generates samples of paths from the process at the specified time points.\n\n    Args:\n      times: Rank 1 `Tensor` of increasing positive real values. The times at\n        which the path points are to be evaluated.\n      num_samples: Positive scalar `int`. The number of paths to draw.\n      initial_state: `Tensor` of shape `[dim]`. The initial state of the\n        process.\n        Default value: None which maps to a zero initial state.\n      random_type: Enum value of `RandomType`. The type of (quasi)-random number\n        generator to use to generate the paths.\n        Default value: None which maps to the standard pseudo-random numbers.\n      seed: Python `int`. The random seed to use. If not supplied, no seed is\n        set.\n      swap_memory: Whether GPU-CPU memory swap is enabled for this op. See\n        equivalent flag in `tf.while_loop` documentation for more details.\n        Useful when computing a gradient of the op since `tf.while_loop` is used\n        to propagate stochastic process in time.\n      name: str. The name to give this op. If not supplied, default name of\n        `sample_paths` is used.\n      **kwargs: parameters, specific to Euler schema: `grid_step` is rank 0 real\n        `Tensor` - maximal distance between points in grid in Euler schema. Note\n        that Euler sampling is only used if it is not possible to do exact\n        sampling because total drift or total covariance are unavailable.\n\n    Returns:\n     A real `Tensor` of shape [num_samples, k, n] where `k` is the size of the\n        `times`, `n` is the dimension of the process.\n    '
        if self._total_drift_fn is None or self._total_covariance_fn is None:
            return super(BrownianMotion, self).sample_paths(times, num_samples=num_samples, initial_state=initial_state, random_type=random_type, seed=seed, name=name, **kwargs)
        default_name = self._name + '_sample_path'
        with tf.compat.v1.name_scope(name, default_name=default_name, values=[times, initial_state]):
            end_times = tf.convert_to_tensor(times, dtype=self.dtype())
            start_times = tf.concat([tf.zeros([1], dtype=end_times.dtype), end_times[:-1]], axis=0)
            paths = self._exact_sampling(end_times, start_times, num_samples, initial_state, random_type, seed)
            if initial_state is not None:
                return paths + initial_state
            return paths

    def _exact_sampling(self, end_times, start_times, num_samples, initial_state, random_type, seed):
        if False:
            for i in range(10):
                print('nop')
        'Returns a sample of paths from the process.'
        non_decreasing = tf.debugging.assert_greater_equal(end_times, start_times, message='Sampling times must be non-decreasing')
        starts_non_negative = tf.debugging.assert_greater_equal(start_times, tf.zeros_like(start_times), message='Sampling times must not be < 0.')
        with tf.compat.v1.control_dependencies([starts_non_negative, non_decreasing]):
            drifts = self._total_drift_fn(start_times, end_times)
            covars = self._total_covariance_fn(start_times, end_times)
            path_deltas = mvn.multivariate_normal((num_samples,), mean=drifts, covariance_matrix=covars, random_type=random_type, seed=seed)
            paths = tf.cumsum(path_deltas, axis=1)
        return paths

    def fd_solver_backward(self, final_time, discounting_fn=None, grid_spec=None, time_step=None, time_step_fn=None, values_batch_size=1, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Returns a solver for solving Feynman-Kac PDE associated to the process.\n\n    Represents the PDE\n\n    ```None\n      V_t + Sum[a_i(t) V_i, 1<=i<=n] +\n        (1/2) Sum[ D_{ij}(t) V_{ij}, 1 <= i,j <= n] - r(t, x) V = 0\n    ```\n\n    In the above, `V_t` is the derivative of `V` with respect to `t`,\n    `V_i` is the partial derivative with respect to `x_i` and `V_{ij}` the\n    (mixed) partial derivative with respect to `x_i` and `x_j`. `D_{ij}` are\n    the components of the diffusion tensor:\n\n    ```None\n      D_{ij}(t) = (Sigma . Transpose[Sigma])_{ij}(t)\n    ```\n\n    This method provides a finite difference solver to solve the above\n    differential equation. Whereas the coefficients `mu` and `D` are properties\n    of the SDE itself, the function `r(t, x)` may be arbitrarily specified\n    by the user (the parameter `discounting_fn` to this method).\n\n    Args:\n      final_time: Positive scalar real `Tensor`. The time of the final value.\n        The solver is initialized to this final time.\n      discounting_fn: Python callable corresponding to the function `r(t, x)` in\n        the description above. The callable accepts two positional arguments.\n        The first argument is the time at which the discount rate function is\n        needed. The second argument contains the values of the state at which\n        the discount is to be computed.\n        Default value: None which maps to `r(t, x) = 0`.\n      grid_spec: An iterable convertible to a tuple containing at least the\n        attributes named 'grid', 'dim' and 'sizes'. For a full description of\n        the fields and expected types, see `grids.GridSpec` which provides the\n        canonical specification of this object.\n      time_step: A real positive scalar `Tensor` or None. The fixed\n        discretization parameter along the time dimension. Either this argument\n        or the `time_step_fn` must be specified. It is an error to specify both.\n        Default value: None.\n      time_step_fn: A callable accepting an instance of `grids.GridStepperState`\n        and returning the size of the next time step as a real scalar tensor.\n        This argument allows usage of a non-constant time step while stepping\n        back. If not specified, the `time_step` parameter must be specified. It\n        is an error to specify both.\n        Default value: None.\n      values_batch_size: A positive Python int. The batch size of values to be\n        propagated simultaneously.\n        Default value: 1.\n      name: Python str. The name to give this op.\n        Default value: None which maps to `fd_solver_backward`.\n      **kwargs: Any other keyword args needed.\n\n    Returns:\n      An instance of `BackwardGridStepper` configured for solving the\n      Feynman-Kac PDE associated to this process.\n    "
        raise NotImplementedError('Finite difference solver not implemented')

def _prefer_static_shape(tensor):
    if False:
        print('Hello World!')
    'Returns the static shape if fully specified else the dynamic shape.'
    tensor = tf.convert_to_tensor(tensor)
    static_shape = tensor.shape
    if static_shape.is_fully_defined():
        return static_shape
    return tf.shape(tensor)

def _prefer_static_rank(tensor):
    if False:
        for i in range(10):
            print('nop')
    'Returns the static rank if fully specified else the dynamic rank.'
    tensor = tf.convert_to_tensor(tensor)
    if tensor.shape.rank is None:
        return tf.rank(tensor)
    return tensor.shape.rank