"""Ops for computing common window functions."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

def _check_params(window_length, dtype):
    if False:
        while True:
            i = 10
    'Check window_length and dtype params.\n\n  Args:\n    window_length: A scalar value or `Tensor`.\n    dtype: The data type to produce. Must be a floating point type.\n\n  Returns:\n    window_length converted to a tensor of type int32.\n\n  Raises:\n    ValueError: If `dtype` is not a floating point type or window_length is not\n      a scalar.\n  '
    if not dtype.is_floating:
        raise ValueError('dtype must be a floating point type. Found %s' % dtype)
    window_length = ops.convert_to_tensor(window_length, dtype=dtypes.int32)
    window_length.shape.assert_has_rank(0)
    return window_length

@tf_export('signal.kaiser_window')
@dispatch.add_dispatch_support
def kaiser_window(window_length, beta=12.0, dtype=dtypes.float32, name=None):
    if False:
        i = 10
        return i + 15
    'Generate a [Kaiser window][kaiser].\n\n  Args:\n    window_length: A scalar `Tensor` indicating the window length to generate.\n    beta: Beta parameter for Kaiser window, see reference below.\n    dtype: The data type to produce. Must be a floating point type.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of shape `[window_length]` of type `dtype`.\n\n  [kaiser]:\n    https://docs.scipy.org/doc/numpy/reference/generated/numpy.kaiser.html\n  '
    with ops.name_scope(name, 'kaiser_window'):
        window_length = _check_params(window_length, dtype)
        window_length_const = tensor_util.constant_value(window_length)
        if window_length_const == 1:
            return array_ops.ones([1], dtype=dtype)
        halflen_float = (math_ops.cast(window_length, dtype=dtypes.float32) - 1.0) / 2.0
        arg = math_ops.range(-halflen_float, halflen_float + 0.1, dtype=dtypes.float32)
        arg = math_ops.cast(arg, dtype=dtype)
        beta = math_ops.cast(beta, dtype=dtype)
        one = math_ops.cast(1.0, dtype=dtype)
        halflen_float = math_ops.cast(halflen_float, dtype=dtype)
        num = beta * math_ops.sqrt(nn_ops.relu(one - math_ops.square(arg / halflen_float)))
        window = math_ops.exp(num - beta) * (special_math_ops.bessel_i0e(num) / special_math_ops.bessel_i0e(beta))
    return window

@tf_export('signal.kaiser_bessel_derived_window')
@dispatch.add_dispatch_support
def kaiser_bessel_derived_window(window_length, beta=12.0, dtype=dtypes.float32, name=None):
    if False:
        return 10
    'Generate a [Kaiser Bessel derived window][kbd].\n\n  Args:\n    window_length: A scalar `Tensor` indicating the window length to generate.\n    beta: Beta parameter for Kaiser window.\n    dtype: The data type to produce. Must be a floating point type.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of shape `[window_length]` of type `dtype`.\n\n  [kbd]:\n    https://en.wikipedia.org/wiki/Kaiser_window#Kaiser%E2%80%93Bessel-derived_(KBD)_window\n  '
    with ops.name_scope(name, 'kaiser_bessel_derived_window'):
        window_length = _check_params(window_length, dtype)
        halflen = window_length // 2
        kaiserw = kaiser_window(halflen + 1, beta, dtype=dtype)
        kaiserw_csum = math_ops.cumsum(kaiserw)
        halfw = math_ops.sqrt(kaiserw_csum[:-1] / kaiserw_csum[-1])
        window = array_ops.concat((halfw, halfw[::-1]), axis=0)
    return window

@tf_export('signal.vorbis_window')
@dispatch.add_dispatch_support
def vorbis_window(window_length, dtype=dtypes.float32, name=None):
    if False:
        print('Hello World!')
    'Generate a [Vorbis power complementary window][vorbis].\n\n  Args:\n    window_length: A scalar `Tensor` indicating the window length to generate.\n    dtype: The data type to produce. Must be a floating point type.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of shape `[window_length]` of type `dtype`.\n\n  [vorbis]:\n    https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform#Window_functions\n  '
    with ops.name_scope(name, 'vorbis_window'):
        window_length = _check_params(window_length, dtype)
        arg = math_ops.cast(math_ops.range(window_length), dtype=dtype)
        window = math_ops.sin(np.pi / 2.0 * math_ops.pow(math_ops.sin(np.pi / math_ops.cast(window_length, dtype=dtype) * (arg + 0.5)), 2.0))
    return window

@tf_export('signal.hann_window')
@dispatch.add_dispatch_support
def hann_window(window_length, periodic=True, dtype=dtypes.float32, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Generate a [Hann window][hann].\n\n  Args:\n    window_length: A scalar `Tensor` indicating the window length to generate.\n    periodic: A bool `Tensor` indicating whether to generate a periodic or\n      symmetric window. Periodic windows are typically used for spectral\n      analysis while symmetric windows are typically used for digital\n      filter design.\n    dtype: The data type to produce. Must be a floating point type.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of shape `[window_length]` of type `dtype`.\n\n  Raises:\n    ValueError: If `dtype` is not a floating point type.\n\n  [hann]: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows\n  '
    return _raised_cosine_window(name, 'hann_window', window_length, periodic, dtype, 0.5, 0.5)

@tf_export('signal.hamming_window')
@dispatch.add_dispatch_support
def hamming_window(window_length, periodic=True, dtype=dtypes.float32, name=None):
    if False:
        return 10
    'Generate a [Hamming][hamming] window.\n\n  Args:\n    window_length: A scalar `Tensor` indicating the window length to generate.\n    periodic: A bool `Tensor` indicating whether to generate a periodic or\n      symmetric window. Periodic windows are typically used for spectral\n      analysis while symmetric windows are typically used for digital\n      filter design.\n    dtype: The data type to produce. Must be a floating point type.\n    name: An optional name for the operation.\n\n  Returns:\n    A `Tensor` of shape `[window_length]` of type `dtype`.\n\n  Raises:\n    ValueError: If `dtype` is not a floating point type.\n\n  [hamming]:\n    https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows\n  '
    return _raised_cosine_window(name, 'hamming_window', window_length, periodic, dtype, 0.54, 0.46)

def _raised_cosine_window(name, default_name, window_length, periodic, dtype, a, b):
    if False:
        i = 10
        return i + 15
    'Helper function for computing a raised cosine window.\n\n  Args:\n    name: Name to use for the scope.\n    default_name: Default name to use for the scope.\n    window_length: A scalar `Tensor` or integer indicating the window length.\n    periodic: A bool `Tensor` indicating whether to generate a periodic or\n      symmetric window.\n    dtype: A floating point `DType`.\n    a: The alpha parameter to the raised cosine window.\n    b: The beta parameter to the raised cosine window.\n\n  Returns:\n    A `Tensor` of shape `[window_length]` of type `dtype`.\n\n  Raises:\n    ValueError: If `dtype` is not a floating point type or `window_length` is\n      not scalar or `periodic` is not scalar.\n  '
    if not dtype.is_floating:
        raise ValueError('dtype must be a floating point type. Found %s' % dtype)
    with ops.name_scope(name, default_name, [window_length, periodic]):
        window_length = ops.convert_to_tensor(window_length, dtype=dtypes.int32, name='window_length')
        window_length.shape.assert_has_rank(0)
        window_length_const = tensor_util.constant_value(window_length)
        if window_length_const == 1:
            return array_ops.ones([1], dtype=dtype)
        periodic = math_ops.cast(ops.convert_to_tensor(periodic, dtype=dtypes.bool, name='periodic'), dtypes.int32)
        periodic.shape.assert_has_rank(0)
        even = 1 - math_ops.mod(window_length, 2)
        n = math_ops.cast(window_length + periodic * even - 1, dtype=dtype)
        count = math_ops.cast(math_ops.range(window_length), dtype)
        cos_arg = constant_op.constant(2 * np.pi, dtype=dtype) * count / n
        if window_length_const is not None:
            return math_ops.cast(a - b * math_ops.cos(cos_arg), dtype=dtype)
        return cond.cond(math_ops.equal(window_length, 1), lambda : array_ops.ones([window_length], dtype=dtype), lambda : math_ops.cast(a - b * math_ops.cos(cos_arg), dtype=dtype))