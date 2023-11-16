"""Initializers for TF 2."""
import math
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops.init_ops import _compute_fans
from tensorflow.python.util.tf_export import tf_export
_PARTITION_SHAPE = 'partition_shape'
_PARTITION_OFFSET = 'partition_offset'

class Initializer:
    """Initializer base class: all initializers inherit from this class.

  Initializers should implement a `__call__` method with the following
  signature:

  ```python
  def __call__(self, shape, dtype=None, **kwargs):
    # returns a tensor of shape `shape` and dtype `dtype`
    # containing values drawn from a distribution of your choice.
  ```
  """

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. If not provided will return tensor\n        of `tf.float32`.\n      **kwargs: Additional keyword arguments. Accepted values:\n        `partition_shape` and `partition_offset`. Used when creating a single\n        partition in a partitioned variable. `partition_shape` is the shape of\n        the partition (i.e. the shape of the returned tensor) and\n        `partition_offset` is a tuple of `int` specifying the offset of this\n        partition w.r.t each axis. For example, a tensor of shape `(30, 100)`\n        can be partitioned into two partitions: `p0` of shape `(10, 100)` and\n        `p1` of shape `(20, 100)`; if the initializer is called with\n        `partition_shape=(20, 100)` and `partition_offset=(10, 0)`, it should\n        return the value for `p1`.\n    '
        raise NotImplementedError

    def get_config(self):
        if False:
            print('Hello World!')
        'Returns the configuration of the initializer as a JSON-serializable dict.\n\n    Returns:\n      A JSON-serializable Python dict.\n    '
        return {}

    @classmethod
    def from_config(cls, config):
        if False:
            print('Hello World!')
        'Instantiates an initializer from a configuration dictionary.\n\n    Example:\n\n    ```python\n    initializer = RandomUniform(-1, 1)\n    config = initializer.get_config()\n    initializer = RandomUniform.from_config(config)\n    ```\n\n    Args:\n      config: A Python dictionary.\n        It will typically be the output of `get_config`.\n\n    Returns:\n      An Initializer instance.\n    '
        config.pop('dtype', None)
        return cls(**config)

    def _validate_kwargs(self, kwargs, support_partition=True):
        if False:
            return 10
        for kwarg in kwargs:
            if kwarg not in [_PARTITION_SHAPE, _PARTITION_OFFSET]:
                raise TypeError(f'Keyword argument should be one of {list([_PARTITION_SHAPE, _PARTITION_OFFSET])}. Received: {kwarg}')
            elif not support_partition:
                raise ValueError(f"{self.__class__.__name__} initializer doesn't support partition-related arguments")

@tf_export('zeros_initializer', v1=[])
class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.zeros_initializer())
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([0., 0., 0.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
  """

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        if False:
            return 10
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are\n       supported.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      ValuesError: If the dtype is not numeric or boolean.\n    '
        self._validate_kwargs(kwargs)
        dtype = dtypes.as_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == dtypes.string:
            raise ValueError(f'Argument `dtype` expected to be numeric or boolean. Received {dtype}.')
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return array_ops.zeros(shape, dtype)

@tf_export('ones_initializer', v1=[])
class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.ones_initializer())
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([1., 1., 1.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...
  """

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are\n        supported.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      ValuesError: If the dtype is not numeric or boolean.\n    '
        self._validate_kwargs(kwargs)
        dtype = dtypes.as_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == dtypes.string:
            raise ValueError(f'Argument `dtype` expected to be numeric or boolean. Received {dtype}.')
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return array_ops.ones(shape, dtype)

@tf_export('constant_initializer', v1=[])
class Constant(Initializer):
    """Initializer that generates tensors with constant values.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  `tf.constant_initializer` returns an object which when called returns a tensor
  populated with the `value` specified in the constructor. This `value` must be
  convertible to the requested `dtype`.

  The argument `value` can be a scalar constant value, or a list of
  values. Scalars broadcast to whichever shape is requested from the
  initializer.

  If `value` is a list, then the length of the list must be equal to the number
  of elements implied by the desired shape of the tensor. If the total number of
  elements in `value` is not equal to the number of elements required by the
  tensor shape, the initializer will raise a `TypeError`.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.constant_initializer(2.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([2., 2., 2.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
  >>> init = tf.constant_initializer(value)
  >>> # Fitting shape
  >>> tf.Variable(init(shape=[2, 4], dtype=tf.float32))
  <tf.Variable ...
  array([[0., 1., 2., 3.],
         [4., 5., 6., 7.]], dtype=float32)>
  >>> # Larger shape
  >>> tf.Variable(init(shape=[3, 4], dtype=tf.float32))
  Traceback (most recent call last):
  ...
  TypeError: ...value has 8 elements, shape is (3, 4) with 12 elements...
  >>> # Smaller shape
  >>> tf.Variable(init(shape=[2, 3], dtype=tf.float32))
  Traceback (most recent call last):
  ...
  TypeError: ...value has 8 elements, shape is (2, 3) with 6 elements...

  Args:
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy
      array. All elements of the initialized variable will be set to the
      corresponding value in the `value` argument.
    support_partition: If true, the initizer supports passing partition
        offset and partition shape arguments to variable creators. This is
        particularly useful when initializing sharded variables where each
        variable shard is initialized to a slice of constant initializer.
      
  Raises:
    TypeError: If the input `value` is not one of the expected types.
  """

    def __init__(self, value=0, support_partition=False):
        if False:
            for i in range(10):
                print('nop')
        if not (np.isscalar(value) or isinstance(value, (list, tuple, np.ndarray))):
            raise TypeError(f'Invalid type for initial value: {type(value).__name__}. Expected Python scalar, list or tuple of values, or numpy.ndarray.')
        self.value = value
        self.support_partition = support_partition

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            while True:
                i = 10
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. If not provided the dtype of the\n        tensor created will be the type of the inital value.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      TypeError: If the initializer cannot create a tensor of the requested\n       dtype.\n    '
        self._validate_kwargs(kwargs, support_partition=self.support_partition)
        if dtype is not None:
            dtype = dtypes.as_dtype(dtype)
        return constant_op.constant(self.value, dtype=dtype, shape=shape)

    def get_config(self):
        if False:
            return 10
        return {'value': self.value}

@tf_export('random_uniform_initializer', v1=[])
class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.ones_initializer())
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([1., 1., 1.], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  array([[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]], dtype=float32)>
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate (inclusive).
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate (exclusive).
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
  """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        if False:
            while True:
                i = 10
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point and integer\n        types are supported.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      ValueError: If the dtype is not numeric.\n    '
        self._validate_kwargs(kwargs)
        dtype = dtypes.as_dtype(dtype)
        if not dtype.is_floating and (not dtype.is_integer):
            raise ValueError(f'Argument `dtype` expected to be numeric or boolean. Received {dtype}.')
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return self._random_generator.random_uniform(shape, self.minval, self.maxval, dtype)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return {'minval': self.minval, 'maxval': self.maxval, 'seed': self.seed}

@tf_export('random_normal_initializer', v1=[])
class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3,
  ...                         tf.random_normal_initializer(mean=1., stddev=2.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([...], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  ...
  >>> make_variables(4, tf.random_uniform_initializer(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

  """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        if False:
            print('Hello World!')
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n        supported.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      ValueError: If the dtype is not floating point\n    '
        self._validate_kwargs(kwargs)
        dtype = _assert_float_dtype(dtype)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return self._random_generator.random_normal(shape, self.mean, self.stddev, dtype)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return {'mean': self.mean, 'stddev': self.stddev, 'seed': self.seed}

class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  These values are similar to values from a `tf.initializers.RandomNormal`
  except that values more than two standard deviations from the mean are
  discarded and re-drawn. This is the recommended initializer for neural network
  weights and filters.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(
  ...     3, tf.initializers.TruncatedNormal(mean=1., stddev=2.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([...], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  ...
  >>> make_variables(4, tf.initializers.RandomUniform(minval=-1., maxval=1.))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
  """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        if False:
            print('Hello World!')
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        if False:
            while True:
                i = 10
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n        supported.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      ValueError: If the dtype is not floating point\n    '
        self._validate_kwargs(kwargs)
        dtype = _assert_float_dtype(dtype)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return self._random_generator.truncated_normal(shape, self.mean, self.stddev, dtype)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return {'mean': self.mean, 'stddev': self.stddev, 'seed': self.seed}

class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights tensors.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  With `distribution="truncated_normal" or "untruncated_normal"`, samples are
  drawn from a truncated/untruncated normal distribution with a mean of zero and
  a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`
  where n is:

    - number of input units in the weight tensor, if mode = "fan_in"
    - number of output units, if mode = "fan_out"
    - average of the numbers of input and output units, if mode = "fan_avg"

  With `distribution="uniform"`, samples are drawn from a uniform distribution
  within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.VarianceScaling(scale=1.))
  >>> v1
  <tf.Variable ... shape=(3,) ... numpy=array([...], dtype=float32)>
  >>> v2
  <tf.Variable ... shape=(3, 3) ... numpy=
  ...
  >>> make_variables(4, tf.initializers.VarianceScaling(distribution='uniform'))
  (<tf.Variable...shape=(4,) dtype=float32...>, <tf.Variable...shape=(4, 4) ...

  Args:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "truncated_normal",
      "untruncated_normal" and  "uniform".
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

  Raises:
    ValueError: In case of an invalid value for the "scale", mode" or
      "distribution" arguments.
  """

    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None):
        if False:
            print('Hello World!')
        if scale <= 0.0:
            raise ValueError(f'Argument `scale` must be a positive float. Received: {scale}')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError(f"Argument `mode` should be one of ('fan_in', 'fan_out', 'fan_avg'). Received: {mode}")
        distribution = distribution.lower()
        if distribution == 'normal':
            distribution = 'truncated_normal'
        if distribution not in {'uniform', 'truncated_normal', 'untruncated_normal'}:
            raise ValueError(f"Argument `distribution` should be one of ('uniform', 'truncated_normal', 'untruncated_normal'). Received: {distribution}")
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        if False:
            return 10
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n        supported.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      ValueError: If the dtype is not floating point\n    '
        self._validate_kwargs(kwargs)
        dtype = _assert_float_dtype(dtype)
        scale = self.scale
        (fan_in, fan_out) = _compute_fans(shape)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        if self.mode == 'fan_in':
            scale /= max(1.0, fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == 'truncated_normal':
            stddev = math.sqrt(scale) / 0.8796256610342398
            return self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)
        elif self.distribution == 'untruncated_normal':
            stddev = math.sqrt(scale)
            return self._random_generator.random_normal(shape, 0.0, stddev, dtype)
        else:
            limit = math.sqrt(3.0 * scale)
            return self._random_generator.random_uniform(shape, -limit, limit, dtype)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return {'scale': self.scale, 'mode': self.mode, 'distribution': self.distribution, 'seed': self.seed}

class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  If the shape of the tensor to initialize is two-dimensional, it is initialized
  with an orthogonal matrix obtained from the QR decomposition of a matrix of
  random numbers drawn from a normal distribution.
  If the matrix has fewer rows than columns then the output will have orthogonal
  rows. Otherwise, the output will have orthogonal columns.

  If the shape of the tensor to initialize is more than two-dimensional,
  a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
  is initialized, where `n` is the length of the shape vector.
  The matrix is subsequently reshaped to give a tensor of the desired shape.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.Orthogonal())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.Orthogonal(gain=0.5))
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

  References:
      [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
      ([pdf](https://arxiv.org/pdf/1312.6120.pdf))
  """

    def __init__(self, gain=1.0, seed=None):
        if False:
            return 10
        self.gain = gain
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        if False:
            return 10
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n        supported.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      ValueError: If the dtype is not floating point or the input shape is not\n       valid.\n    '
        self._validate_kwargs(kwargs, support_partition=False)
        dtype = _assert_float_dtype(dtype)
        if len(shape) < 2:
            raise ValueError(f'The tensor to initialize, specified by argument `shape` must be at least two-dimensional. Received shape={shape}')
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))
        a = self._random_generator.random_normal(flat_shape, dtype=dtype)
        (q, r) = gen_linalg_ops.qr(a, full_matrices=False)
        d = array_ops.diag_part(r)
        q *= math_ops.sign(d)
        if num_rows < num_cols:
            q = array_ops.matrix_transpose(q)
        return self.gain * array_ops.reshape(q, shape)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return {'gain': self.gain, 'seed': self.seed}

class Identity(Initializer):
    """Initializer that generates the identity matrix.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Only usable for generating 2D matrices.

  Examples:

  >>> def make_variable(k, initializer):
  ...   return tf.Variable(initializer(shape=[k, k], dtype=tf.float32))
  >>> make_variable(2, tf.initializers.Identity())
  <tf.Variable ... shape=(2, 2) dtype=float32, numpy=
  array([[1., 0.],
         [0., 1.]], dtype=float32)>
  >>> make_variable(3, tf.initializers.Identity(gain=0.5))
  <tf.Variable ... shape=(3, 3) dtype=float32, numpy=
  array([[0.5, 0. , 0. ],
         [0. , 0.5, 0. ],
         [0. , 0. , 0.5]], dtype=float32)>

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
  """

    def __init__(self, gain=1.0):
        if False:
            while True:
                i = 10
        self.gain = gain

    def __call__(self, shape, dtype=dtypes.float32, **kwargs):
        if False:
            return 10
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n       supported.\n      **kwargs: Additional keyword arguments.\n\n    Raises:\n      ValueError: If the dtype is not floating point\n      ValueError: If the requested shape does not have exactly two axes.\n    '
        self._validate_kwargs(kwargs, support_partition=False)
        dtype = _assert_float_dtype(dtype)
        if len(shape) != 2:
            raise ValueError(f'The tensor to initialize, specified by argument `shape` must be at least two-dimensional. Received shape={shape}')
        initializer = linalg_ops_impl.eye(*shape, dtype=dtype)
        return self.gain * initializer

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return {'gain': self.gain}

class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a uniform distribution within [-limit, limit] where `limit`
  is `sqrt(6 / (fan_in + fan_out))` where `fan_in` is the number of input units
  in the weight tensor and `fan_out` is the number of output units in the weight
  tensor.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.GlorotUniform())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.RandomNormal())
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

    def __init__(self, seed=None):
        if False:
            return 10
        super(GlorotUniform, self).__init__(scale=1.0, mode='fan_avg', distribution='uniform', seed=seed)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return {'seed': self.seed}

class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a truncated normal distribution centered on 0 with `stddev
  = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units in
  the weight tensor and `fan_out` is the number of output units in the weight
  tensor.

  Examples:

  >>> def make_variables(k, initializer):
  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))
  >>> v1, v2 = make_variables(3, tf.initializers.GlorotNormal())
  >>> v1
  <tf.Variable ... shape=(3, 3) ...
  >>> v2
  <tf.Variable ... shape=(3, 3, 3) ...
  >>> make_variables(4, tf.initializers.RandomNormal())
  (<tf.Variable ... shape=(4, 4) dtype=float32...
   <tf.Variable ... shape=(4, 4, 4) dtype=float32...

  Args:
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.

  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

    def __init__(self, seed=None):
        if False:
            for i in range(10):
                print('nop')
        super(GlorotNormal, self).__init__(scale=1.0, mode='fan_avg', distribution='truncated_normal', seed=seed)

    def get_config(self):
        if False:
            return 10
        return {'seed': self.seed}
zeros_initializer = Zeros
ones_initializer = Ones
constant_initializer = Constant
random_uniform_initializer = RandomUniform
random_normal_initializer = RandomNormal
truncated_normal_initializer = TruncatedNormal
variance_scaling_initializer = VarianceScaling
glorot_uniform_initializer = GlorotUniform
glorot_normal_initializer = GlorotNormal
orthogonal_initializer = Orthogonal
identity_initializer = Identity

def lecun_normal(seed=None):
    if False:
        print('Hello World!')
    'LeCun normal initializer.\n\n  Initializers allow you to pre-specify an initialization strategy, encoded in\n  the Initializer object, without knowing the shape and dtype of the variable\n  being initialized.\n\n  Draws samples from a truncated normal distribution centered on 0 with `stddev\n  = sqrt(1 / fan_in)` where `fan_in` is the number of input units in the weight\n  tensor.\n\n  Examples:\n\n  >>> def make_variables(k, initializer):\n  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),\n  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))\n  >>> v1, v2 = make_variables(3, tf.initializers.lecun_normal())\n  >>> v1\n  <tf.Variable ... shape=(3, 3) ...\n  >>> v2\n  <tf.Variable ... shape=(3, 3, 3) ...\n  >>> make_variables(4, tf.initializers.RandomNormal())\n  (<tf.Variable ... shape=(4, 4) dtype=float32...\n   <tf.Variable ... shape=(4, 4, 4) dtype=float32...\n\n  Args:\n    seed: A Python integer. Used to seed the random generator.\n\n  Returns:\n    A callable Initializer with `shape` and `dtype` arguments which generates a\n    tensor.\n\n  References:\n      - Self-Normalizing Neural Networks,\n      [Klambauer et al., 2017]\n      (https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)\n      ([pdf]\n      (https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))\n      - Efficient Backprop,\n      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)\n  '
    return VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal', seed=seed)

def lecun_uniform(seed=None):
    if False:
        for i in range(10):
            print('nop')
    'LeCun uniform initializer.\n\n  Initializers allow you to pre-specify an initialization strategy, encoded in\n  the Initializer object, without knowing the shape and dtype of the variable\n  being initialized.\n\n  Draws samples from a uniform distribution within [-limit, limit] where `limit`\n  is `sqrt(3 / fan_in)` where `fan_in` is the number of input units in the\n  weight tensor.\n\n  Examples:\n\n  >>> def make_variables(k, initializer):\n  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),\n  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))\n  >>> v1, v2 = make_variables(3, tf.initializers.lecun_uniform())\n  >>> v1\n  <tf.Variable ... shape=(3, 3) ...\n  >>> v2\n  <tf.Variable ... shape=(3, 3, 3) ...\n  >>> make_variables(4, tf.initializers.RandomNormal())\n  (<tf.Variable ... shape=(4, 4) dtype=float32...\n   <tf.Variable ... shape=(4, 4, 4) dtype=float32...\n\n  Args:\n    seed: A Python integer. Used to seed the random generator.\n\n  Returns:\n    A callable Initializer with `shape` and `dtype` arguments which generates a\n    tensor.\n\n  References:\n      - Self-Normalizing Neural Networks,\n      [Klambauer et al., 2017](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks) # pylint: disable=line-too-long\n      ([pdf](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))\n      - Efficient Backprop,\n      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)\n  '
    return VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=seed)

def he_normal(seed=None):
    if False:
        return 10
    'He normal initializer.\n\n  Initializers allow you to pre-specify an initialization strategy, encoded in\n  the Initializer object, without knowing the shape and dtype of the variable\n  being initialized.\n\n  It draws samples from a truncated normal distribution centered on 0 with\n  `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in the\n  weight tensor.\n\n  Examples:\n\n  >>> def make_variables(k, initializer):\n  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),\n  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))\n  >>> v1, v2 = make_variables(3, tf.initializers.he_normal())\n  >>> v1\n  <tf.Variable ... shape=(3, 3) ...\n  >>> v2\n  <tf.Variable ... shape=(3, 3, 3) ...\n  >>> make_variables(4, tf.initializers.RandomNormal())\n  (<tf.Variable ... shape=(4, 4) dtype=float32...\n   <tf.Variable ... shape=(4, 4, 4) dtype=float32...\n\n  Args:\n    seed: A Python integer. Used to seed the random generator.\n\n  Returns:\n    A callable Initializer with `shape` and `dtype` arguments which generates a\n    tensor.\n\n  References:\n      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long\n      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))\n  '
    return VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=seed)

def he_uniform(seed=None):
    if False:
        return 10
    'He uniform variance scaling initializer.\n\n  Initializers allow you to pre-specify an initialization strategy, encoded in\n  the Initializer object, without knowing the shape and dtype of the variable\n  being initialized.\n\n  Draws samples from a uniform distribution within [-limit, limit] where `limit`\n  is `sqrt(6 / fan_in)` where `fan_in` is the number of input units in the\n  weight tensor.\n\n  Examples:\n\n  >>> def make_variables(k, initializer):\n  ...   return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),\n  ...           tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))\n  >>> v1, v2 = make_variables(3, tf.initializers.he_uniform())\n  >>> v1\n  <tf.Variable ... shape=(3, 3) ...\n  >>> v2\n  <tf.Variable ... shape=(3, 3, 3) ...\n  >>> make_variables(4, tf.initializers.RandomNormal())\n  (<tf.Variable ... shape=(4, 4) dtype=float32...\n   <tf.Variable ... shape=(4, 4, 4) dtype=float32...\n\n  Args:\n    seed: A Python integer. Used to seed the random generator.\n\n  Returns:\n    A callable Initializer with `shape` and `dtype` arguments which generates a\n    tensor.\n\n  References:\n      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long\n      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))\n  '
    return VarianceScaling(scale=2.0, mode='fan_in', distribution='uniform', seed=seed)

def _assert_float_dtype(dtype):
    if False:
        return 10
    'Validate and return floating point type based on `dtype`.\n\n  `dtype` must be a floating point type.\n\n  Args:\n    dtype: The data type to validate.\n\n  Returns:\n    Validated type.\n\n  Raises:\n    ValueError: if `dtype` is not a floating point type.\n  '
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError(f'Argument `dtype` is expected to be floating point. Received: {dtype}.')
    return dtype

class _RandomGenerator:
    """Random generator that selects appropriate random ops."""

    def __init__(self, seed=None):
        if False:
            return 10
        super(_RandomGenerator, self).__init__()
        if seed is not None:
            self.seed = [seed, 0]
        else:
            self.seed = None

    def random_normal(self, shape, mean=0.0, stddev=1, dtype=dtypes.float32):
        if False:
            i = 10
            return i + 15
        'A deterministic random normal if seed is passed.'
        if self.seed:
            op = stateless_random_ops.stateless_random_normal
        else:
            op = random_ops.random_normal
        return op(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)

    def random_uniform(self, shape, minval, maxval, dtype):
        if False:
            print('Hello World!')
        'A deterministic random uniform if seed is passed.'
        if self.seed:
            op = stateless_random_ops.stateless_random_uniform
        else:
            op = random_ops.random_uniform
        return op(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=self.seed)

    def truncated_normal(self, shape, mean, stddev, dtype):
        if False:
            print('Hello World!')
        'A deterministic truncated normal if seed is passed.'
        if self.seed:
            op = stateless_random_ops.stateless_truncated_normal
        else:
            op = random_ops.truncated_normal
        return op(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)
zero = zeros = Zeros
one = ones = Ones
constant = Constant
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
identity = Identity
orthogonal = Orthogonal
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform