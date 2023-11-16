"""Random functions."""
import numpy as onp
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
DEFAULT_RANDN_DTYPE = onp.float32

@tf_export.tf_export('experimental.numpy.random.seed', v1=[])
@np_utils.np_doc('random.seed')
def seed(s):
    if False:
        return 10
    'Sets the seed for the random number generator.\n\n  Uses `tf.set_random_seed`.\n\n  Args:\n    s: an integer.\n  '
    try:
        s = int(s)
    except TypeError:
        raise ValueError(f'Argument `s` got an invalid value {s}. Only integers are supported.')
    random_seed.set_seed(s)

@tf_export.tf_export('experimental.numpy.random.randn', v1=[])
@np_utils.np_doc('random.randn')
def randn(*args):
    if False:
        while True:
            i = 10
    'Returns samples from a normal distribution.\n\n  Uses `tf.random_normal`.\n\n  Args:\n    *args: The shape of the output array.\n\n  Returns:\n    An ndarray with shape `args` and dtype `float64`.\n  '
    return standard_normal(size=args)

@tf_export.tf_export('experimental.numpy.random.standard_normal', v1=[])
@np_utils.np_doc('random.standard_normal')
def standard_normal(size=None):
    if False:
        for i in range(10):
            print('nop')
    if size is None:
        size = ()
    elif np_utils.isscalar(size):
        size = (size,)
    dtype = np_utils.result_type(float)
    return random_ops.random_normal(size, dtype=dtype)

@tf_export.tf_export('experimental.numpy.random.uniform', v1=[])
@np_utils.np_doc('random.uniform')
def uniform(low=0.0, high=1.0, size=None):
    if False:
        i = 10
        return i + 15
    dtype = np_utils.result_type(float)
    low = np_array_ops.asarray(low, dtype=dtype)
    high = np_array_ops.asarray(high, dtype=dtype)
    if size is None:
        size = array_ops.broadcast_dynamic_shape(low.shape, high.shape)
    return random_ops.random_uniform(shape=size, minval=low, maxval=high, dtype=dtype)

@tf_export.tf_export('experimental.numpy.random.poisson', v1=[])
@np_utils.np_doc('random.poisson')
def poisson(lam=1.0, size=None):
    if False:
        while True:
            i = 10
    if size is None:
        size = ()
    elif np_utils.isscalar(size):
        size = (size,)
    return random_ops.random_poisson(shape=size, lam=lam, dtype=np_dtypes.int_)

@tf_export.tf_export('experimental.numpy.random.random', v1=[])
@np_utils.np_doc('random.random')
def random(size=None):
    if False:
        i = 10
        return i + 15
    return uniform(0.0, 1.0, size)

@tf_export.tf_export('experimental.numpy.random.rand', v1=[])
@np_utils.np_doc('random.rand')
def rand(*size):
    if False:
        return 10
    return uniform(0.0, 1.0, size)

@tf_export.tf_export('experimental.numpy.random.randint', v1=[])
@np_utils.np_doc('random.randint')
def randint(low, high=None, size=None, dtype=onp.int64):
    if False:
        return 10
    low = int(low)
    if high is None:
        high = low
        low = 0
    if size is None:
        size = ()
    elif isinstance(size, int):
        size = (size,)
    dtype_orig = dtype
    dtype = np_utils.result_type(dtype)
    accepted_dtypes = (onp.int32, onp.int64)
    if dtype not in accepted_dtypes:
        raise ValueError(f'Argument `dtype` got an invalid value {dtype_orig}. Only those convertible to {accepted_dtypes} are supported.')
    return random_ops.random_uniform(shape=size, minval=low, maxval=high, dtype=dtype)