import enum
from functools import partial
import numpy as np
from numpy import asarray
from numpy.fft._pocketfft import _cook_nd_args, _raw_fft, _raw_fftnd

class NormMode(enum.Enum):
    none = 1
    by_sqrt_n = 2
    by_n = 3

def _get_norm_mode(norm, forward):
    if False:
        print('Hello World!')
    if norm == 'ortho':
        return NormMode.by_sqrt_n
    if norm is None or norm == 'backward':
        return NormMode.none if forward else NormMode.by_n
    return NormMode.by_n if forward else NormMode.none

def _get_inv_norm(n, norm_mode):
    if False:
        return 10
    assert isinstance(norm_mode, NormMode), f'invalid norm_type {norm_mode}'
    if norm_mode == NormMode.none:
        return 1.0
    if norm_mode == NormMode.by_sqrt_n:
        return np.sqrt(n)
    return n

def _fftc2c(a, n=None, axis=-1, norm=None, forward=None):
    if False:
        i = 10
        return i + 15
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_inv_norm(n, norm)
    output = _raw_fft(a, n, axis, False, forward, inv_norm)
    return output

def _fftr2c(a, n=None, axis=-1, norm=None, forward=None):
    if False:
        while True:
            i = 10
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_inv_norm(n, norm)
    output = _raw_fft(a, n, axis, True, True, inv_norm)
    if not forward:
        output = output.conj()
    return output

def _fftc2r(a, n=None, axis=-1, norm=None, forward=None):
    if False:
        return 10
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    inv_norm = _get_inv_norm(n, norm)
    output = _raw_fft(a.conj() if forward else a, n, axis, True, False, inv_norm)
    return output

def _fft_c2c_nd(x, axes, norm_mode, forward):
    if False:
        for i in range(10):
            print('nop')
    f = partial(_fftc2c, forward=forward)
    y = _raw_fftnd(x, s=None, axes=axes, function=f, norm=norm_mode)
    return y

def _fft_r2c_nd(x, axes, norm_mode, forward, onesided):
    if False:
        print('Hello World!')
    a = asarray(x)
    (s, axes) = _cook_nd_args(a, axes=axes)
    if onesided:
        a = _fftr2c(a, s[-1], axes[-1], norm_mode, forward)
        a = _fft_c2c_nd(a, axes[:-1], norm_mode, forward)
    else:
        a = _fft_c2c_nd(x, axes, norm_mode, forward)
    return a

def _fft_c2r_nd(x, axes, norm_mode, forward, last_dim_size):
    if False:
        i = 10
        return i + 15
    a = asarray(x)
    (s, axes) = _cook_nd_args(a, axes=axes, invreal=1)
    if last_dim_size is not None:
        s[-1] = last_dim_size
    a = _fft_c2c_nd(a, axes[:-1], norm_mode, forward)
    a = _fftc2r(a, s[-1], axes[-1], norm_mode, forward)
    return a

def fft_c2c(x, axes, normalization, forward):
    if False:
        while True:
            i = 10
    norm_mode = _get_norm_mode(normalization, forward)
    return _fft_c2c_nd(x, axes, norm_mode, forward)

def fft_c2r(x, axes, normalization, forward, last_dim_size):
    if False:
        for i in range(10):
            print('nop')
    norm_mode = _get_norm_mode(normalization, forward)
    return _fft_c2r_nd(x, axes, norm_mode, forward, last_dim_size)

def fft_r2c(x, axes, normalization, forward, onesided):
    if False:
        print('Hello World!')
    norm_mode = _get_norm_mode(normalization, forward)
    return _fft_r2c_nd(x, axes, norm_mode, forward, onesided)

def fft_c2c_backward(dy, axes, normalization, forward):
    if False:
        i = 10
        return i + 15
    norm_mode = _get_norm_mode(normalization, forward)
    dx = _fft_c2c_nd(dy, axes, norm_mode, not forward)
    return dx

def fft_r2c_backward(x, dy, axes, normalization, forward, onesided):
    if False:
        print('Hello World!')
    a = dy
    if not onesided:
        a = fft_c2c_backward(a, axes, normalization, forward)
    else:
        pad_widths = [(0, 0)] * a.ndim
        last_axis = axes[-1]
        if last_axis < 0:
            last_axis += a.ndim
        last_dim_size = a.shape[last_axis]
        pad_widths[last_axis] = (0, x.shape[last_axis] - last_dim_size)
        a = np.pad(a, pad_width=pad_widths)
        a = fft_c2c_backward(a, axes, normalization, forward)
    return a.real

def _fft_fill_conj_grad(x, axes, length_to_double):
    if False:
        print('Hello World!')
    last_fft_axis = axes[-1]
    shape = x.shape
    for multi_index in np.ndindex(*shape):
        if 0 < multi_index[last_fft_axis] and multi_index[last_fft_axis] <= length_to_double:
            x[multi_index] *= 2
    return x

def fft_c2r_backward(x, dy, axes, normalization, forward, last_dim_size):
    if False:
        for i in range(10):
            print('nop')
    norm_mode = _get_norm_mode(normalization, forward)
    a = dy
    a = _fft_r2c_nd(dy, axes, norm_mode, not forward, True)
    last_fft_axis = axes[-1]
    length_to_double = dy.shape[last_fft_axis] - x.shape[last_fft_axis]
    a = _fft_fill_conj_grad(a, axes, length_to_double)
    return a