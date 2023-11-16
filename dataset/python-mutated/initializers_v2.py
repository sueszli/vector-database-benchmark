"""Keras initializers for TF 2."""
import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
_PARTITION_SHAPE = 'partition_shape'
_PARTITION_OFFSET = 'partition_offset'

class Initializer(object):
    """Initializer base class: all Keras initializers inherit from this class.

  Initializers should implement a `__call__` method with the following
  signature:

  ```python
  def __call__(self, shape, dtype=None, **kwargs):
    # returns a tensor of shape `shape` and dtype `dtype`
    # containing values drawn from a distribution of your choice.
  ```

  Optionally, you an also implement the method `get_config` and the class
  method `from_config` in order to support serialization -- just like with
  any Keras object.

  Here's a simple example: a random normal initializer.

  ```python
  import tensorflow as tf

  class ExampleRandomNormal(tf.keras.initializers.Initializer):

    def __init__(self, mean, stddev):
      self.mean = mean
      self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
      return tf.random.normal(
          shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

    def get_config(self):  # To support serialization
      return {"mean": self.mean, "stddev": self.stddev}
  ```

  Note that we don't have to implement `from_config` in the example above since
  the constructor arguments of the class the keys in the config returned by
  `get_config` are the same. In this case, the default `from_config`
  works fine.
  """

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor.\n      **kwargs: Additional keyword arguments.\n    '
        raise NotImplementedError

    def get_config(self):
        if False:
            while True:
                i = 10
        'Returns the configuration of the initializer as a JSON-serializable dict.\n\n    Returns:\n      A JSON-serializable Python dict.\n    '
        return {}

    @classmethod
    def from_config(cls, config):
        if False:
            while True:
                i = 10
        'Instantiates an initializer from a configuration dictionary.\n\n    Example:\n\n    ```python\n    initializer = RandomUniform(-1, 1)\n    config = initializer.get_config()\n    initializer = RandomUniform.from_config(config)\n    ```\n\n    Args:\n      config: A Python dictionary, the output of `get_config`.\n\n    Returns:\n      A `tf.keras.initializers.Initializer` instance.\n    '
        config.pop('dtype', None)
        return cls(**config)

class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

  Also available via the shortcut function `tf.keras.initializers.zeros`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Zeros()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Zeros()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
  """

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            print('Hello World!')
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are\n       supported. If not specified, `tf.keras.backend.floatx()` is used,\n       which default to `float32` unless you configured it otherwise\n       (via `tf.keras.backend.set_floatx(float_dtype)`).\n      **kwargs: Additional keyword arguments.\n    '
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == dtypes.string:
            raise ValueError('Expected numeric or boolean dtype, got %s.' % dtype)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return array_ops.zeros(shape, dtype)

class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

  Also available via the shortcut function `tf.keras.initializers.ones`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Ones()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Ones()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
  """

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            return 10
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are\n       supported. If not specified, `tf.keras.backend.floatx()` is used,\n       which default to `float32` unless you configured it otherwise\n       (via `tf.keras.backend.set_floatx(float_dtype)`).\n      **kwargs: Additional keyword arguments.\n    '
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_numpy_compatible or dtype == dtypes.string:
            raise ValueError('Expected numeric or boolean dtype, got %s.' % dtype)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return array_ops.ones(shape, dtype)

class Constant(Initializer):
    """Initializer that generates tensors with constant values.

  Also available via the shortcut function `tf.keras.initializers.constant`.

  Only scalar values are allowed.
  The constant value provided must be convertible to the dtype requested
  when calling the initializer.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Constant(3.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Constant(3.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    value: A Python scalar.
  """

    def __init__(self, value=0):
        if False:
            return 10
        self.value = value

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            return 10
        'Returns a tensor object initialized to `self.value`.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. If not specified,\n       `tf.keras.backend.floatx()` is used,\n       which default to `float32` unless you configured it otherwise\n       (via `tf.keras.backend.set_floatx(float_dtype)`).\n      **kwargs: Additional keyword arguments.\n    '
        del kwargs
        return constant_op.constant(self.value, dtype=_get_dtype(dtype), shape=shape)

    def get_config(self):
        if False:
            while True:
                i = 10
        return {'value': self.value}

class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

  Also available via the shortcut function
  `tf.keras.initializers.random_uniform`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate (inclusive).
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate (exclusive).
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        if False:
            print('Hello World!')
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            print('Hello World!')
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point and integer\n      types are supported. If not specified,\n        `tf.keras.backend.floatx()` is used,\n       which default to `float32` unless you configured it otherwise\n       (via `tf.keras.backend.set_floatx(float_dtype)`).\n      **kwargs: Additional keyword arguments.\n    '
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _get_dtype(dtype)
        if not dtype.is_floating and (not dtype.is_integer):
            raise ValueError('Expected float or integer dtype, got %s.' % dtype)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return self._random_generator.random_uniform(shape, self.minval, self.maxval, dtype)

    def get_config(self):
        if False:
            print('Hello World!')
        return {'minval': self.minval, 'maxval': self.maxval, 'seed': self.seed}

class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

  Also available via the shortcut function
  `tf.keras.initializers.random_normal`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate.
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        if False:
            i = 10
            return i + 15
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            while True:
                i = 10
        'Returns a tensor object initialized to random normal values.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n        supported. If not specified, `tf.keras.backend.floatx()` is used, which\n        default to `float32` unless you configured it otherwise (via\n        `tf.keras.backend.set_floatx(float_dtype)`)\n      **kwargs: Additional keyword arguments.\n    '
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return self._random_generator.random_normal(shape, self.mean, self.stddev, dtype)

    def get_config(self):
        if False:
            return 10
        return {'mean': self.mean, 'stddev': self.stddev, 'seed': self.seed}

class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

  Also available via the shortcut function
  `tf.keras.initializers.truncated_normal`.

  The values generated are similar to values from a
  `tf.keras.initializers.RandomNormal` initializer except that values more
  than two standard deviations from the mean are
  discarded and re-drawn.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate before truncation.
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        if False:
            return 10
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            print('Hello World!')
        'Returns a tensor object initialized to random normal values (truncated).\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n        supported. If not specified, `tf.keras.backend.floatx()` is used, which\n        default to `float32` unless you configured it otherwise (via\n        `tf.keras.backend.set_floatx(float_dtype)`)\n      **kwargs: Additional keyword arguments.\n    '
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        return self._random_generator.truncated_normal(shape, self.mean, self.stddev, dtype)

    def get_config(self):
        if False:
            print('Hello World!')
        return {'mean': self.mean, 'stddev': self.stddev, 'seed': self.seed}

class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights tensors.

  Also available via the shortcut function
  `tf.keras.initializers.variance_scaling`.

  With `distribution="truncated_normal" or "untruncated_normal"`, samples are
  drawn from a truncated/untruncated normal distribution with a mean of zero and
  a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`,
  where `n` is:

  - number of input units in the weight tensor, if `mode="fan_in"`
  - number of output units, if `mode="fan_out"`
  - average of the numbers of input and output units, if `mode="fan_avg"`

  With `distribution="uniform"`, samples are drawn from a uniform distribution
  within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.VarianceScaling(
  ... scale=0.1, mode='fan_in', distribution='uniform')
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.VarianceScaling(
  ... scale=0.1, mode='fan_in', distribution='uniform')
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "truncated_normal",
      "untruncated_normal" and  "uniform".
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None):
        if False:
            print('Hello World!')
        if scale <= 0.0:
            raise ValueError('`scale` must be positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument:', mode)
        distribution = distribution.lower()
        if distribution == 'normal':
            distribution = 'truncated_normal'
        if distribution not in {'uniform', 'truncated_normal', 'untruncated_normal'}:
            raise ValueError('Invalid `distribution` argument:', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            return 10
        'Returns a tensor object initialized as specified by the initializer.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n        supported. If not specified, `tf.keras.backend.floatx()` is used, which\n        default to `float32` unless you configured it otherwise (via\n        `tf.keras.backend.set_floatx(float_dtype)`)\n      **kwargs: Additional keyword arguments.\n    '
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
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
            while True:
                i = 10
        return {'scale': self.scale, 'mode': self.mode, 'distribution': self.distribution, 'seed': self.seed}

class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix.

  Also available via the shortcut function `tf.keras.initializers.orthogonal`.

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

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Orthogonal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Orthogonal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

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

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns a tensor object initialized to an orthogonal matrix.\n\n    Args:\n      shape: Shape of the tensor.\n      dtype: Optional dtype of the tensor. Only floating point types are\n        supported. If not specified, `tf.keras.backend.floatx()` is used,\n       which default to `float32` unless you configured it otherwise\n       (via `tf.keras.backend.set_floatx(float_dtype)`)\n      **kwargs: Additional keyword arguments.\n    '
        _validate_kwargs(self.__class__.__name__, kwargs, support_partition=False)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if len(shape) < 2:
            raise ValueError('The tensor to initialize must be at least two-dimensional')
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))
        a = self._random_generator.random_normal(flat_shape, dtype=dtype)
        (q, r) = gen_linalg_ops.qr(a, full_matrices=False)
        d = array_ops.tensor_diag_part(r)
        q *= math_ops.sign(d)
        if num_rows < num_cols:
            q = array_ops.matrix_transpose(q)
        return self.gain * array_ops.reshape(q, shape)

    def get_config(self):
        if False:
            return 10
        return {'gain': self.gain, 'seed': self.seed}

class Identity(Initializer):
    """Initializer that generates the identity matrix.

  Also available via the shortcut function `tf.keras.initializers.identity`.

  Only usable for generating 2D matrices.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Identity()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Identity()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
  """

    def __init__(self, gain=1.0):
        if False:
            return 10
        self.gain = gain

    def __call__(self, shape, dtype=None, **kwargs):
        if False:
            while True:
                i = 10
        'Returns a tensor object initialized to a 2D identity matrix.\n\n    Args:\n      shape: Shape of the tensor. It should have exactly rank 2.\n      dtype: Optional dtype of the tensor. Only floating point types are\n       supported. If not specified, `tf.keras.backend.floatx()` is used,\n       which default to `float32` unless you configured it otherwise\n       (via `tf.keras.backend.set_floatx(float_dtype)`)\n      **kwargs: Additional keyword arguments.\n    '
        _validate_kwargs(self.__class__.__name__, kwargs, support_partition=False)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if len(shape) != 2:
            raise ValueError('Identity matrix initializer can only be used for 2D matrices.')
        initializer = linalg_ops.eye(*shape, dtype=dtype)
        return self.gain * initializer

    def get_config(self):
        if False:
            return 10
        return {'gain': self.gain}

class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

  Also available via the shortcut function
  `tf.keras.initializers.glorot_uniform`.

  Draws samples from a uniform distribution within `[-limit, limit]`, where
  `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input units
  in the weight tensor and `fan_out` is the number of output units).

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.GlorotUniform()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.GlorotUniform()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
      ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """

    def __init__(self, seed=None):
        if False:
            i = 10
            return i + 15
        super(GlorotUniform, self).__init__(scale=1.0, mode='fan_avg', distribution='uniform', seed=seed)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return {'seed': self.seed}

class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer.

  Also available via the shortcut function
  `tf.keras.initializers.glorot_normal`.

  Draws samples from a truncated normal distribution centered on 0 with `stddev
  = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units in
  the weight tensor and `fan_out` is the number of output units in the weight
  tensor.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.GlorotNormal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.GlorotNormal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

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

class LecunNormal(VarianceScaling):
    """Lecun normal initializer.

   Also available via the shortcut function
  `tf.keras.initializers.lecun_normal`.

  Initializers allow you to pre-specify an initialization strategy, encoded in
  the Initializer object, without knowing the shape and dtype of the variable
  being initialized.

  Draws samples from a truncated normal distribution centered on 0 with `stddev
  = sqrt(1 / fan_in)` where `fan_in` is the number of input units in the weight
  tensor.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.LecunNormal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.LecunNormal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. Used to seed the random generator.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al., 2017]
      (https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)
      ([pdf]
      (https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """

    def __init__(self, seed=None):
        if False:
            i = 10
            return i + 15
        super(LecunNormal, self).__init__(scale=1.0, mode='fan_in', distribution='truncated_normal', seed=seed)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        return {'seed': self.seed}

class LecunUniform(VarianceScaling):
    """Lecun uniform initializer.

   Also available via the shortcut function
  `tf.keras.initializers.lecun_uniform`.

  Draws samples from a uniform distribution within `[-limit, limit]`,
  where `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
  weight tensor).

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.LecunUniform()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.LecunUniform()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al., 2017](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks) # pylint: disable=line-too-long
      ([pdf](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  """

    def __init__(self, seed=None):
        if False:
            i = 10
            return i + 15
        super(LecunUniform, self).__init__(scale=1.0, mode='fan_in', distribution='uniform', seed=seed)

    def get_config(self):
        if False:
            while True:
                i = 10
        return {'seed': self.seed}

class HeNormal(VarianceScaling):
    """He normal initializer.

   Also available via the shortcut function
  `tf.keras.initializers.he_normal`.

  It draws samples from a truncated normal distribution centered on 0 with
  `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in the
  weight tensor.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.HeNormal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.HeNormal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """

    def __init__(self, seed=None):
        if False:
            for i in range(10):
                print('nop')
        super(HeNormal, self).__init__(scale=2.0, mode='fan_in', distribution='truncated_normal', seed=seed)

    def get_config(self):
        if False:
            while True:
                i = 10
        return {'seed': self.seed}

class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer.

   Also available via the shortcut function
  `tf.keras.initializers.he_uniform`.

  Draws samples from a uniform distribution within `[-limit, limit]`, where
  `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
  weight tensor).

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.HeUniform()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.HeUniform()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [He et al., 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  """

    def __init__(self, seed=None):
        if False:
            for i in range(10):
                print('nop')
        super(HeUniform, self).__init__(scale=2.0, mode='fan_in', distribution='uniform', seed=seed)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        return {'seed': self.seed}

def _get_dtype(dtype):
    if False:
        while True:
            i = 10
    if dtype is None:
        dtype = backend.floatx()
    return dtypes.as_dtype(dtype)

def _assert_float_dtype(dtype):
    if False:
        return 10
    'Validate and return floating point type based on `dtype`.\n\n  `dtype` must be a floating point type.\n\n  Args:\n    dtype: The data type to validate.\n\n  Returns:\n    Validated type.\n\n  Raises:\n    ValueError: if `dtype` is not a floating point type.\n  '
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError('Expected floating point type, got %s.' % dtype)
    return dtype

class _RandomGenerator(object):
    """Random generator that selects appropriate random ops."""

    def __init__(self, seed=None):
        if False:
            for i in range(10):
                print('nop')
        super(_RandomGenerator, self).__init__()
        if seed is not None:
            self.seed = [seed, 0]
        else:
            self.seed = None

    def random_normal(self, shape, mean=0.0, stddev=1, dtype=dtypes.float32):
        if False:
            return 10
        'A deterministic random normal if seed is passed.'
        if self.seed:
            op = stateless_random_ops.stateless_random_normal
        else:
            op = random_ops.random_normal
        return op(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=self.seed)

    def random_uniform(self, shape, minval, maxval, dtype):
        if False:
            return 10
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

def _compute_fans(shape):
    if False:
        while True:
            i = 10
    'Computes the number of input and output units for a weight shape.\n\n  Args:\n    shape: Integer shape tuple or TF tensor shape.\n\n  Returns:\n    A tuple of integer scalars (fan_in, fan_out).\n  '
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return (int(fan_in), int(fan_out))

def _validate_kwargs(cls_name, kwargs, support_partition=True):
    if False:
        print('Hello World!')
    for kwarg in kwargs:
        if kwarg not in [_PARTITION_SHAPE, _PARTITION_OFFSET]:
            raise TypeError('Unknown keyword arguments: %s' % kwarg)
        elif not support_partition:
            raise ValueError("%s initializer doesn't support partition-related arguments" % cls_name)