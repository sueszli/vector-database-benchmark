"""Real-to-real transforms

cuFFT does not implement real-to-real FFTs. This module implements forward
and inverse DCT-II and DCT-III transforms using FFTs.

A length N DCT can be computed using a length N FFT and some additional
multiplications and reordering of entries.

The approach taken here is based on the work in [1]_, [2]_ and is discussed in
the freely-available online resources [3]_, [4]_.

The implementation here follows that approach with only minor modification to
match the normalization conventions in SciPy.

The modifications to turn a type II or III DCT to a DST were implemented as
described in [5]_.

.. [1] J. Makhoul, "A fast cosine transform in one and two dimensions," in
    IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 28,
    no. 1, pp. 27-34, February 1980.

.. [2] M.J. Narasimha and A.M. Peterson, “On the computation of the discrete
    cosine  transform,” IEEE Trans. Commun., vol. 26, no. 6, pp. 934–936, 1978.

.. [3] http://fourier.eng.hmc.edu/e161/lectures/dct/node2.html

.. [4] https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft  # noqa

.. [5] X. Shao, S. G. Johnson. Type-II/III DCT/DST algorithms with reduced
    number of arithmetic operations, Signal Processing, Volume 88, Issue 6,
    pp. 1553-1564, 2008.
"""
import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
__all__ = ['dct', 'dctn', 'dst', 'dstn', 'idct', 'idctn', 'idst', 'idstn']

def _promote_dtype(x):
    if False:
        print('Hello World!')
    if x.dtype.kind in 'bui':
        float_dtype = cupy.float64
    else:
        float_dtype = cupy.promote_types(x.dtype, cupy.float32)
    return x.astype(float_dtype, copy=False)

def _get_dct_norm_factor(n, inorm, dct_type=2):
    if False:
        for i in range(10):
            print('nop')
    "Normalization factors for DCT/DST I-IV.\n\n    Parameters\n    ----------\n    n : int\n        Data size.\n    inorm : {'none', 'sqrt', 'full'}\n        When `inorm` is 'none', the scaling factor is 1.0 (unnormalized). When\n        `inorm` is 1, scaling by ``1/sqrt(d)`` as needed for an orthogonal\n        transform is used. When `inorm` is 2, normalization by ``1/d`` is\n        applied. The value of ``d`` depends on both `n` and the `dct_type`.\n    dct_type : {1, 2, 3, 4}\n        Which type of DCT or DST is being normalized?.\n\n    Returns\n    -------\n    fct : float\n        The normalization factor.\n    "
    if inorm == 'none':
        return 1
    delta = -1 if dct_type == 1 else 0
    d = 2 * (n + delta)
    if inorm == 'full':
        fct = 1 / d
    elif inorm == 'sqrt':
        fct = 1 / math.sqrt(d)
    else:
        raise ValueError('expected inorm = "none", "sqrt" or "full"')
    return fct

def _reshuffle_dct2(x, n, axis, dst=False):
    if False:
        while True:
            i = 10
    'Reorder entries to allow computation of DCT/DST-II via FFT.'
    sl_even = [slice(None)] * x.ndim
    sl_even[axis] = slice(0, None, 2)
    sl_even = tuple(sl_even)
    sl_odd = [slice(None)] * x.ndim
    if n % 2:
        sl_odd[axis] = slice(-2, None, -2)
        sl_odd = tuple(sl_odd)
    else:
        sl_odd[axis] = slice(None, None, -2)
        sl_odd = tuple(sl_odd)
    if dst:
        x = cupy.concatenate((x[sl_even], -x[sl_odd]), axis=axis)
    else:
        x = cupy.concatenate((x[sl_even], x[sl_odd]), axis=axis)
    return x
_mult_factor_dct2 = _core.ElementwiseKernel(in_params='R xr, int32 N, R norm_factor', out_params='C y', operation='\n    C j(0., -1.);\n    y = (R)2.0 * norm_factor * exp(j * (R)(i * M_PI / (2 * N)));')

def _exp_factor_dct2(x, n, axis, norm_factor, n_truncate=None):
    if False:
        return 10
    'Twiddle & scaling factors for computation of DCT/DST-II via FFT.'
    if n_truncate is None:
        n_truncate = n
    tmp = cupy.empty((n_truncate,), dtype=x.dtype)
    _mult_factor_dct2(tmp.real, n, norm_factor, tmp)
    if x.ndim == 1:
        return tmp
    tmp_shape = [1] * x.ndim
    tmp_shape[axis] = n_truncate
    tmp_shape = tuple(tmp_shape)
    return tmp.reshape(tmp_shape)

def _dct_or_dst_type2(x, n=None, axis=-1, forward=True, norm=None, dst=False, overwrite_x=False):
    if False:
        i = 10
        return i + 15
    "Forward DCT/DST-II (or inverse DCT/DST-III) along a single axis\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The data to transform.\n    n : int\n        The size of the transform. If None, ``x.shape[axis]`` is used.\n    axis : int\n        Axis along which the transform is applied.\n    forward : bool\n        Set true to indicate that this is a forward DCT-II as opposed to an\n        inverse DCT-III (The difference between the two is only in the\n        normalization factor).\n    norm : {None, 'ortho', 'forward', 'backward'}\n        The normalization convention to use.\n    dst : bool\n        If True, a discrete sine transform is computed rather than the discrete\n        cosine transform.\n    overwrite_x : bool\n        Indicates that it is okay to overwrite x. In practice, the current\n        implementation never performs the transform in-place.\n\n    Returns\n    -------\n    y: cupy.ndarray\n        The transformed array.\n    "
    if axis < -x.ndim or axis >= x.ndim:
        raise numpy.AxisError('axis out of range')
    if axis < 0:
        axis += x.ndim
    if n is not None and n < 1:
        raise ValueError(f'invalid number of data points ({n}) specified')
    x = _cook_shape(x, (n,), (axis,), 'R2R')
    n = x.shape[axis]
    x = _reshuffle_dct2(x, x.shape[axis], axis, dst)
    if norm == 'ortho':
        inorm = 'sqrt'
    elif norm == 'forward':
        inorm = 'full' if forward else 'none'
    else:
        inorm = 'none' if forward else 'full'
    norm_factor = _get_dct_norm_factor(n, inorm=inorm, dct_type=2)
    x = _fft.fft(x, n=n, axis=axis, overwrite_x=True)
    tmp = _exp_factor_dct2(x, n, axis, norm_factor)
    x *= tmp
    x = cupy.real(x)
    if dst:
        slrev = [slice(None)] * x.ndim
        slrev[axis] = slice(None, None, -1)
        x = x[tuple(slrev)]
    if norm == 'ortho':
        sl0 = [slice(None)] * x.ndim
        sl0[axis] = slice(1)
        x[tuple(sl0)] *= math.sqrt(2) * 0.5
    return x

def _reshuffle_dct3(y, n, axis, dst):
    if False:
        i = 10
        return i + 15
    'Reorder entries to allow computation of DCT/DST-II via FFT.'
    x = cupy.empty_like(y)
    n_half = (n + 1) // 2
    sl_even = [slice(None)] * y.ndim
    sl_even[axis] = slice(0, None, 2)
    sl_even = tuple(sl_even)
    sl_half = [slice(None)] * y.ndim
    sl_half[axis] = slice(0, n_half)
    x[sl_even] = y[tuple(sl_half)]
    sl_odd = [slice(None)] * y.ndim
    sl_odd[axis] = slice(1, None, 2)
    sl_odd = tuple(sl_odd)
    sl_half[axis] = slice(-1, n_half - 1, -1)
    if dst:
        x[sl_odd] = -y[tuple(sl_half)]
    else:
        x[sl_odd] = y[tuple(sl_half)]
    return x
_mult_factor_dct3 = _core.ElementwiseKernel(in_params='R xr, int32 N, R norm_factor', out_params='C y', operation='\n    C j(0., 1.);\n    y = (R)(2 * N * norm_factor) * exp(j * (R)(i * M_PI / (2 * N)));')

def _exp_factor_dct3(x, n, axis, dtype, norm_factor):
    if False:
        for i in range(10):
            print('nop')
    'Twiddle & scaling factors for computation of DCT/DST-III via FFT.'
    tmp = cupy.empty((n,), dtype=dtype)
    _mult_factor_dct3(tmp.real, n, norm_factor, tmp)
    if x.ndim == 1:
        return tmp
    tmp_shape = [1] * x.ndim
    tmp_shape[axis] = n
    tmp_shape = tuple(tmp_shape)
    return tmp.reshape(tmp_shape)

def _dct_or_dst_type3(x, n=None, axis=-1, norm=None, forward=True, dst=False, overwrite_x=False):
    if False:
        print('Hello World!')
    "Forward DCT/DST-III (or inverse DCT/DST-II) along a single axis.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The data to transform.\n    n : int\n        The size of the transform. If None, ``x.shape[axis]`` is used.\n    axis : int\n        Axis along which the transform is applied.\n    forward : bool\n        Set true to indicate that this is a forward DCT-II as opposed to an\n        inverse DCT-III (The difference between the two is only in the\n        normalization factor).\n    norm : {None, 'ortho', 'forward', 'backward'}\n        The normalization convention to use.\n    dst : bool\n        If True, a discrete sine transform is computed rather than the discrete\n        cosine transform.\n    overwrite_x : bool\n        Indicates that it is okay to overwrite x. In practice, the current\n        implementation never performs the transform in-place.\n\n    Returns\n    -------\n    y: cupy.ndarray\n        The transformed array.\n\n    "
    if axis < -x.ndim or axis >= x.ndim:
        raise numpy.AxisError('axis out of range')
    if axis < 0:
        axis += x.ndim
    if n is not None and n < 1:
        raise ValueError(f'invalid number of data points ({n}) specified')
    x = _cook_shape(x, (n,), (axis,), 'R2R')
    n = x.shape[axis]
    if norm == 'ortho':
        sl0_scale = 0.5 * math.sqrt(2)
        inorm = 'sqrt'
    elif norm == 'forward':
        sl0_scale = 0.5
        inorm = 'full' if forward else 'none'
    elif norm == 'backward' or norm is None:
        sl0_scale = 0.5
        inorm = 'none' if forward else 'full'
    else:
        raise ValueError(f'Invalid norm value "{norm}", should be "backward", "ortho" or "forward"')
    norm_factor = _get_dct_norm_factor(n, inorm=inorm, dct_type=3)
    dtype = cupy.promote_types(x, cupy.complex64)
    sl0 = [slice(None)] * x.ndim
    sl0[axis] = slice(1)
    if dst:
        if norm == 'ortho':
            float_dtype = cupy.promote_types(x.dtype, cupy.float32)
            if x.dtype != float_dtype:
                x = x.astype(float_dtype)
            elif not overwrite_x:
                x = x.copy()
            x[tuple(sl0)] *= math.sqrt(2)
            sl0_scale = 0.5
        slrev = [slice(None)] * x.ndim
        slrev[axis] = slice(None, None, -1)
        x = x[tuple(slrev)]
    tmp = _exp_factor_dct3(x, n, axis, dtype, norm_factor)
    x = x * tmp
    x[tuple(sl0)] *= sl0_scale
    x = _fft.ifft(x, n=n, axis=axis, overwrite_x=True)
    x = cupy.real(x)
    return _reshuffle_dct3(x, n, axis, dst)

@_fft._implements(_fft._scipy_fft.dct)
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    if False:
        i = 10
        return i + 15
    'Return the Discrete Cosine Transform of an array, x.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The input array.\n    type : {1, 2, 3, 4}, optional\n        Type of the DCT (see Notes). Default type is 2. Currently CuPy only\n        supports types 2 and 3.\n    n : int, optional:\n        Length of the transform.  If ``n < x.shape[axis]``, `x` is\n        truncated. If ``n > x.shape[axis]``, `x` is zero-padded.\n        The default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the dct is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    norm : {"backward", "ortho", "forward"}, optional\n        Normalization mode (see Notes). Default is "backward".\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : cupy.ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    :func:`scipy.fft.dct`\n\n    Notes\n    -----\n    For a single dimension array ``x``, ``dct(x, norm=\'ortho\')`` is equal\n    to MATLAB ``dct(x)``.\n\n    For ``norm="ortho"`` both the `dct` and `idct` are scaled by the same\n    overall factor in both directions. By default, the transform is also\n    orthogonalized which for types 1, 2 and 3 means the transform definition is\n    modified to give orthogonality of the DCT matrix (see below).\n\n    For ``norm="backward"``, there is no scaling on `dct` and the `idct` is\n    scaled by ``1/N`` where ``N`` is the "logical" size of the DCT. For\n    ``norm="forward"`` the ``1/N`` normalization is applied to the forward\n    `dct` instead and the `idct` is unnormalized.\n\n    CuPy currently only supports DCT types 2 and 3. \'The\' DCT generally\n    refers to DCT type 2, and \'the\' Inverse DCT generally refers to DCT\n    type 3 [1]_. See the :func:`scipy.fft.dct` documentation for a full\n    description of each type.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Discrete cosine transform",\n           https://en.wikipedia.org/wiki/Discrete_cosine_transform\n\n    '
    if x.dtype.kind == 'c':
        out = dct(x.real, type, n, axis, norm, overwrite_x)
        out = out + 1j * dct(x.imag, type, n, axis, norm, overwrite_x)
        return out
    x = _promote_dtype(x)
    if type == 2:
        return _dct_or_dst_type2(x, n=n, axis=axis, norm=norm, forward=True, dst=False)
    elif type == 3:
        return _dct_or_dst_type3(x, n=n, axis=axis, norm=norm, forward=True, dst=False)
    elif type in [1, 4]:
        raise NotImplementedError('Only DCT-II and DCT-III have been implemented.')
    else:
        raise ValueError('invalid DCT type')

@_fft._implements(_fft._scipy_fft.dst)
def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    if False:
        while True:
            i = 10
    'Return the Discrete Sine Transform of an array, x.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The input array.\n    type : {1, 2, 3, 4}, optional\n        Type of the DST (see Notes). Default type is 2.\n    n : int, optional\n        Length of the transform.  If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the dst is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    norm : {"backward", "ortho", "forward"}, optional\n        Normalization mode (see Notes). Default is "backward".\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    dst : cupy.ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    :func:`scipy.fft.dst`\n\n    Notes\n    -----\n\n    For ``norm="ortho"`` both the `dst` and `idst` are scaled by the same\n    overall factor in both directions. By default, the transform is also\n    orthogonalized which for types 2 and 3 means the transform definition is\n    modified to give orthogonality of the DST matrix (see below).\n\n    For ``norm="backward"``, there is no scaling on the `dst` and the `idst` is\n    scaled by ``1/N`` where ``N`` is the "logical" size of the DST.\n\n    See the :func:`scipy.fft.dst` documentation for a full description of each\n    type. CuPy currently only supports DST types 2 and 3.\n    '
    if x.dtype.kind == 'c':
        out = dst(x.real, type, n, axis, norm, overwrite_x)
        out = out + 1j * dst(x.imag, type, n, axis, norm, overwrite_x)
        return out
    x = _promote_dtype(x)
    if type == 2:
        return _dct_or_dst_type2(x, n=n, axis=axis, norm=norm, forward=True, dst=True)
    elif type == 3:
        return _dct_or_dst_type3(x, n=n, axis=axis, norm=norm, forward=True, dst=True)
    elif type in [1, 4]:
        raise NotImplementedError('Only DST-II and DST-III have been implemented.')
    else:
        raise ValueError('invalid DST type')

@_fft._implements(_fft._scipy_fft.idct)
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    if False:
        while True:
            i = 10
    'Return the Inverse Discrete Cosine Transform of an array, x.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The input array.\n    type : {1, 2, 3, 4}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    n : int, optional\n        Length of the transform.  If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the idct is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    norm : {"backward", "ortho", "forward"}, optional\n        Normalization mode (see Notes). Default is "backward".\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    idct : cupy.ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    :func:`scipy.fft.idct`\n\n    Notes\n    -----\n    For a single dimension array `x`, ``idct(x, norm=\'ortho\')`` is equal to\n    MATLAB ``idct(x)``.\n\n    For ``norm="ortho"`` both the `dct` and `idct` are scaled by the same\n    overall factor in both directions. By default, the transform is also\n    orthogonalized which for types 1, 2 and 3 means the transform definition is\n    modified to give orthogonality of the IDCT matrix (see `dct` for the full\n    definitions).\n\n    \'The\' IDCT is the IDCT-II, which is the same as the normalized DCT-III\n    [1]_. See the :func:`scipy.fft.dct` documentation for a full description of\n    each type. CuPy currently only supports DCT types 2 and 3.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Discrete sine transform",\n           https://en.wikipedia.org/wiki/Discrete_sine_transform\n    '
    if x.dtype.kind == 'c':
        out = idct(x.real, type, n, axis, norm, overwrite_x)
        out = out + 1j * idct(x.imag, type, n, axis, norm, overwrite_x)
        return out
    x = _promote_dtype(x)
    if type == 2:
        return _dct_or_dst_type3(x, n=n, axis=axis, norm=norm, forward=False)
    elif type == 3:
        return _dct_or_dst_type2(x, n=n, axis=axis, norm=norm, forward=False)
    elif type in [1, 4]:
        raise NotImplementedError('Only DCT-II and DCT-III have been implemented.')
    else:
        raise ValueError('invalid DCT type')

@_fft._implements(_fft._scipy_fft.idst)
def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    if False:
        while True:
            i = 10
    'Return the Inverse Discrete Sine Transform of an array, x.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The input array.\n    type : {1, 2, 3, 4}, optional\n        Type of the DST (see Notes). Default type is 2.\n    n : int, optional\n        Length of the transform. If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the idst is computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    norm : {"backward", "ortho", "forward"}, optional\n        Normalization mode (see Notes). Default is "backward".\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    idst : cupy.ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    :func:`scipy.fft.idst`\n\n    Notes\n    -----\n    For full details of the DST types and normalization modes, as well as\n    references, see :func:`scipy.fft.dst`.\n    '
    if x.dtype.kind == 'c':
        out = idst(x.real, type, n, axis, norm, overwrite_x)
        out = out + 1j * idst(x.imag, type, n, axis, norm, overwrite_x)
        return out
    x = _promote_dtype(x)
    if type == 2:
        return _dct_or_dst_type3(x, n=n, axis=axis, norm=norm, forward=False, dst=True)
    elif type == 3:
        return _dct_or_dst_type2(x, n=n, axis=axis, norm=norm, forward=False, dst=True)
    elif type in [1, 4]:
        raise NotImplementedError('Only DST-II and DST-III have been implemented.')
    else:
        raise ValueError('invalid DST type')

def _iterable_of_int(x, name=None):
    if False:
        while True:
            i = 10
    'Convert ``x`` to an iterable sequence of int.'
    if isinstance(x, numbers.Number):
        x = (x,)
    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or 'value'
        raise ValueError(f'{name} must be a scalar or iterable of integers') from e
    return x

def _init_nd_shape_and_axes(x, shape, axes):
    if False:
        while True:
            i = 10
    'Handles shape and axes arguments for nd transforms.'
    noshape = shape is None
    noaxes = axes is None
    if not noaxes:
        axes = _iterable_of_int(axes, 'axes')
        axes = [a + x.ndim if a < 0 else a for a in axes]
        if any((a >= x.ndim or a < 0 for a in axes)):
            raise ValueError('axes exceeds dimensionality of input')
        if len(set(axes)) != len(axes):
            raise ValueError('all axes must be unique')
    if not noshape:
        shape = _iterable_of_int(shape, 'shape')
        nshape = len(shape)
        if axes and len(axes) != nshape:
            raise ValueError('when given, axes and shape arguments have to be of the same length')
        if noaxes:
            if nshape > x.ndim:
                raise ValueError('shape requires more axes than are present')
            axes = range(x.ndim - len(shape), x.ndim)
        shape = [x.shape[a] if s == -1 else s for (s, a) in zip(shape, axes)]
    elif noaxes:
        shape = list(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]
    if any((s < 1 for s in shape)):
        raise ValueError(f'invalid number of data points ({shape}) specified')
    return (shape, axes)

@_fft._implements(_fft._scipy_fft.dctn)
def dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False):
    if False:
        i = 10
        return i + 15
    'Compute a multidimensional Discrete Cosine Transform.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The input array.\n    type : {1, 2, 3, 4}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    s : int or array_like of ints or None, optional\n        The shape of the result. If both `s` and `axes` (see below) are None,\n        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is\n        ``numpy.take(x.shape, axes, axis=0)``.\n        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.\n        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length\n        ``s[i]``.\n        If any element of `s` is -1, the size of the corresponding dimension of\n        `x` is used.\n    axes : int or array_like of ints or None, optional\n        Axes over which the DCT is computed. If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n    norm : {"backward", "ortho", "forward"}, optional\n        Normalization mode (see Notes). Default is "backward".\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : cupy.ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    :func:`scipy.fft.dctn`\n\n    Notes\n    -----\n    For full details of the DCT types and normalization modes, as well as\n    references, see `dct`.\n    '
    if x.dtype.kind == 'c':
        out = dctn(x.real, type, s, axes, norm, overwrite_x)
        out = out + 1j * dctn(x.imag, type, s, axes, norm, overwrite_x)
        return out
    (shape, axes) = _init_nd_shape_and_axes(x, s, axes)
    x = _promote_dtype(x)
    if len(axes) == 0:
        return x
    for (n, axis) in zip(shape, axes):
        x = dct(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)
    return x

@_fft._implements(_fft._scipy_fft.idctn)
def idctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False):
    if False:
        for i in range(10):
            print('nop')
    'Compute a multidimensional Discrete Cosine Transform.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The input array.\n    type : {1, 2, 3, 4}, optional\n        Type of the DCT (see Notes). Default type is 2.\n    s : int or array_like of ints or None, optional\n        The shape of the result. If both `s` and `axes` (see below) are None,\n        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is\n        ``numpy.take(x.shape, axes, axis=0)``.\n        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.\n        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length\n        ``s[i]``.\n        If any element of `s` is -1, the size of the corresponding dimension of\n        `x` is used.\n    axes : int or array_like of ints or None, optional\n        Axes over which the IDCT is computed. If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n    norm : {"backward", "ortho", "forward"}, optional\n        Normalization mode (see Notes). Default is "backward".\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : cupy.ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    :func:`scipy.fft.idctn`\n\n    Notes\n    -----\n    For full details of the IDCT types and normalization modes, as well as\n    references, see :func:`scipy.fft.idct`.\n    '
    if x.dtype.kind == 'c':
        out = idctn(x.real, type, s, axes, norm, overwrite_x)
        out = out + 1j * idctn(x.imag, type, s, axes, norm, overwrite_x)
        return out
    (shape, axes) = _init_nd_shape_and_axes(x, s, axes)
    x = _promote_dtype(x)
    if len(axes) == 0:
        return x
    for (n, axis) in zip(shape, axes):
        x = idct(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)
    return x

@_fft._implements(_fft._scipy_fft.dstn)
def dstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False):
    if False:
        while True:
            i = 10
    'Compute a multidimensional Discrete Sine Transform.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The input array.\n    type : {1, 2, 3, 4}, optional\n        Type of the DST (see Notes). Default type is 2.\n    s : int or array_like of ints or None, optional\n        The shape of the result. If both `s` and `axes` (see below) are None,\n        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is\n        ``numpy.take(x.shape, axes, axis=0)``.\n        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.\n        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length\n        ``s[i]``.\n        If any element of `s` is -1, the size of the corresponding dimension of\n        `x` is used.\n    axes : int or array_like of ints or None, optional\n        Axes over which the DST is computed. If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n    norm : {"backward", "ortho", "forward"}, optional\n        Normalization mode (see Notes). Default is "backward".\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : cupy.ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    :func:`scipy.fft.dstn`\n\n    Notes\n    -----\n    For full details of the DST types and normalization modes, as well as\n    references, see :func:`scipy.fft.dst`.\n    '
    if x.dtype.kind == 'c':
        out = dstn(x.real, type, s, axes, norm, overwrite_x)
        out = out + 1j * dstn(x.imag, type, s, axes, norm, overwrite_x)
        return out
    (shape, axes) = _init_nd_shape_and_axes(x, s, axes)
    x = _promote_dtype(x)
    if len(axes) == 0:
        return x
    for (n, axis) in zip(shape, axes):
        x = dst(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)
    return x

@_fft._implements(_fft._scipy_fft.idstn)
def idstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False):
    if False:
        for i in range(10):
            print('nop')
    'Compute a multidimensional Discrete Sine Transform.\n\n    Parameters\n    ----------\n    x : cupy.ndarray\n        The input array.\n    type : {1, 2, 3, 4}, optional\n        Type of the DST (see Notes). Default type is 2.\n    s : int or array_like of ints or None, optional\n        The shape of the result. If both `s` and `axes` (see below) are None,\n        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is\n        ``numpy.take(x.shape, axes, axis=0)``.\n        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.\n        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length\n        ``s[i]``.\n        If any element of `s` is -1, the size of the corresponding dimension of\n        `x` is used.\n    axes : int or array_like of ints or None, optional\n        Axes over which the IDST is computed. If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n    norm : {"backward", "ortho", "forward"}, optional\n        Normalization mode (see Notes). Default is "backward".\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    y : cupy.ndarray of real\n        The transformed input array.\n\n    See Also\n    --------\n    :func:`scipy.fft.idstn`\n\n    Notes\n    -----\n    For full details of the IDST types and normalization modes, as well as\n    references, see :func:`scipy.fft.idst`.\n    '
    if x.dtype.kind == 'c':
        out = idstn(x.real, type, s, axes, norm, overwrite_x)
        out = out + 1j * idstn(x.imag, type, s, axes, norm, overwrite_x)
        return out
    (shape, axes) = _init_nd_shape_and_axes(x, s, axes)
    x = _promote_dtype(x)
    if len(axes) == 0:
        return x
    for (n, axis) in zip(shape, axes):
        x = idst(x, type=type, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x)
    return x