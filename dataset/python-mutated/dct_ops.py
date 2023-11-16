"""Discrete Cosine Transform ops."""
import math as _math
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

def _validate_dct_arguments(input_tensor, dct_type, n, axis, norm):
    if False:
        i = 10
        return i + 15
    'Checks that DCT/IDCT arguments are compatible and well formed.'
    if axis != -1:
        raise NotImplementedError('axis must be -1. Got: %s' % axis)
    if n is not None and n < 1:
        raise ValueError('n should be a positive integer or None')
    if dct_type not in (1, 2, 3, 4):
        raise ValueError('Types I, II, III and IV (I)DCT are supported.')
    if dct_type == 1:
        if norm == 'ortho':
            raise ValueError('Normalization is not supported for the Type-I DCT.')
        if input_tensor.shape[-1] is not None and input_tensor.shape[-1] < 2:
            raise ValueError('Type-I DCT requires the dimension to be greater than one.')
    if norm not in (None, 'ortho'):
        raise ValueError("Unknown normalization. Expected None or 'ortho', got: %s" % norm)

@tf_export('signal.dct', v1=['signal.dct', 'spectral.dct'])
@dispatch.add_dispatch_support
def dct(input, type=2, n=None, axis=-1, norm=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes the 1D [Discrete Cosine Transform (DCT)][dct] of `input`.\n\n  Types I, II, III and IV are supported.\n  Type I is implemented using a length `2N` padded `tf.signal.rfft`.\n  Type II is implemented using a length `2N` padded `tf.signal.rfft`, as\n   described here: [Type 2 DCT using 2N FFT padded (Makhoul)]\n   (https://dsp.stackexchange.com/a/10606).\n  Type III is a fairly straightforward inverse of Type II\n   (i.e. using a length `2N` padded `tf.signal.irfft`).\n   Type IV is calculated through 2N length DCT2 of padded signal and\n  picking the odd indices.\n\n  @compatibility(scipy)\n  Equivalent to [scipy.fftpack.dct]\n   (https://docs.scipy.org/doc/scipy-1.4.0/reference/generated/scipy.fftpack.dct.html)\n   for Type-I, Type-II, Type-III and Type-IV DCT.\n  @end_compatibility\n\n  Args:\n    input: A `[..., samples]` `float32`/`float64` `Tensor` containing the\n      signals to take the DCT of.\n    type: The DCT type to perform. Must be 1, 2, 3 or 4.\n    n: The length of the transform. If length is less than sequence length,\n      only the first n elements of the sequence are considered for the DCT.\n      If n is greater than the sequence length, zeros are padded and then\n      the DCT is computed as usual.\n    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.\n    norm: The normalization to apply. `None` for no normalization or `'ortho'`\n      for orthonormal normalization.\n    name: An optional name for the operation.\n\n  Returns:\n    A `[..., samples]` `float32`/`float64` `Tensor` containing the DCT of\n    `input`.\n\n  Raises:\n    ValueError: If `type` is not `1`, `2`, `3` or `4`, `axis` is\n      not `-1`, `n` is not `None` or greater than 0,\n      or `norm` is not `None` or `'ortho'`.\n    ValueError: If `type` is `1` and `norm` is `ortho`.\n\n  [dct]: https://en.wikipedia.org/wiki/Discrete_cosine_transform\n  "
    _validate_dct_arguments(input, type, n, axis, norm)
    return _dct_internal(input, type, n, axis, norm, name)

def _dct_internal(input, type=2, n=None, axis=-1, norm=None, name=None):
    if False:
        while True:
            i = 10
    "Computes the 1D Discrete Cosine Transform (DCT) of `input`.\n\n  This internal version of `dct` does not perform any validation and accepts a\n  dynamic value for `n` in the form of a rank 0 tensor.\n\n  Args:\n    input: A `[..., samples]` `float32`/`float64` `Tensor` containing the\n      signals to take the DCT of.\n    type: The DCT type to perform. Must be 1, 2, 3 or 4.\n    n: The length of the transform. If length is less than sequence length,\n      only the first n elements of the sequence are considered for the DCT.\n      If n is greater than the sequence length, zeros are padded and then\n      the DCT is computed as usual. Can be an int or rank 0 tensor.\n    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.\n    norm: The normalization to apply. `None` for no normalization or `'ortho'`\n      for orthonormal normalization.\n    name: An optional name for the operation.\n\n  Returns:\n    A `[..., samples]` `float32`/`float64` `Tensor` containing the DCT of\n    `input`.\n  "
    with _ops.name_scope(name, 'dct', [input]):
        input = _ops.convert_to_tensor(input)
        zero = _ops.convert_to_tensor(0.0, dtype=input.dtype)
        seq_len = tensor_shape.dimension_value(input.shape[-1]) or _array_ops.shape(input)[-1]
        if n is not None:

            def truncate_input():
                if False:
                    for i in range(10):
                        print('nop')
                return input[..., 0:n]

            def pad_input():
                if False:
                    i = 10
                    return i + 15
                rank = len(input.shape)
                padding = [[0, 0] for _ in range(rank)]
                padding[rank - 1][1] = n - seq_len
                padding = _ops.convert_to_tensor(padding, dtype=_dtypes.int32)
                return _array_ops.pad(input, paddings=padding)
            input = smart_cond.smart_cond(n <= seq_len, truncate_input, pad_input)
        axis_dim = tensor_shape.dimension_value(input.shape[-1]) or _array_ops.shape(input)[-1]
        axis_dim_float = _math_ops.cast(axis_dim, input.dtype)
        if type == 1:
            dct1_input = _array_ops.concat([input, input[..., -2:0:-1]], axis=-1)
            dct1 = _math_ops.real(fft_ops.rfft(dct1_input))
            return dct1
        if type == 2:
            scale = 2.0 * _math_ops.exp(_math_ops.complex(zero, -_math_ops.range(axis_dim_float) * _math.pi * 0.5 / axis_dim_float))
            dct2 = _math_ops.real(fft_ops.rfft(input, fft_length=[2 * axis_dim])[..., :axis_dim] * scale)
            if norm == 'ortho':
                n1 = 0.5 * _math_ops.rsqrt(axis_dim_float)
                n2 = n1 * _math.sqrt(2.0)
                weights = _array_ops.pad(_array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]], constant_values=n2)
                dct2 *= weights
            return dct2
        elif type == 3:
            if norm == 'ortho':
                n1 = _math_ops.sqrt(axis_dim_float)
                n2 = n1 * _math.sqrt(0.5)
                weights = _array_ops.pad(_array_ops.expand_dims(n1, 0), [[0, axis_dim - 1]], constant_values=n2)
                input *= weights
            else:
                input *= axis_dim_float
            scale = 2.0 * _math_ops.exp(_math_ops.complex(zero, _math_ops.range(axis_dim_float) * _math.pi * 0.5 / axis_dim_float))
            dct3 = _math_ops.real(fft_ops.irfft(scale * _math_ops.complex(input, zero), fft_length=[2 * axis_dim]))[..., :axis_dim]
            return dct3
        elif type == 4:
            dct2 = _dct_internal(input, type=2, n=2 * axis_dim, axis=axis, norm=None)
            dct4 = dct2[..., 1::2]
            if norm == 'ortho':
                dct4 *= _math.sqrt(0.5) * _math_ops.rsqrt(axis_dim_float)
            return dct4

@tf_export('signal.idct', v1=['signal.idct', 'spectral.idct'])
@dispatch.add_dispatch_support
def idct(input, type=2, n=None, axis=-1, norm=None, name=None):
    if False:
        i = 10
        return i + 15
    "Computes the 1D [Inverse Discrete Cosine Transform (DCT)][idct] of `input`.\n\n  Currently Types I, II, III, IV are supported. Type III is the inverse of\n  Type II, and vice versa.\n\n  Note that you must re-normalize by 1/(2n) to obtain an inverse if `norm` is\n  not `'ortho'`. That is:\n  `signal == idct(dct(signal)) * 0.5 / signal.shape[-1]`.\n  When `norm='ortho'`, we have:\n  `signal == idct(dct(signal, norm='ortho'), norm='ortho')`.\n\n  @compatibility(scipy)\n  Equivalent to [scipy.fftpack.idct]\n   (https://docs.scipy.org/doc/scipy-1.4.0/reference/generated/scipy.fftpack.idct.html)\n   for Type-I, Type-II, Type-III and Type-IV DCT.\n  @end_compatibility\n\n  Args:\n    input: A `[..., samples]` `float32`/`float64` `Tensor` containing the\n      signals to take the DCT of.\n    type: The IDCT type to perform. Must be 1, 2, 3 or 4.\n    n: For future expansion. The length of the transform. Must be `None`.\n    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.\n    norm: The normalization to apply. `None` for no normalization or `'ortho'`\n      for orthonormal normalization.\n    name: An optional name for the operation.\n\n  Returns:\n    A `[..., samples]` `float32`/`float64` `Tensor` containing the IDCT of\n    `input`.\n\n  Raises:\n    ValueError: If `type` is not `1`, `2` or `3`, `n` is not `None, `axis` is\n      not `-1`, or `norm` is not `None` or `'ortho'`.\n\n  [idct]:\n  https://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms\n  "
    _validate_dct_arguments(input, type, n, axis, norm)
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return _dct_internal(input, type=inverse_type, n=n, axis=axis, norm=norm, name=name)