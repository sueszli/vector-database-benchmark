import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy import special

def _get_output_fourier(output, input, complex_only=False):
    if False:
        i = 10
        return i + 15
    types = [cupy.complex64, cupy.complex128]
    if not complex_only:
        types += [cupy.float32, cupy.float64]
    if output is None:
        if input.dtype in types:
            output = cupy.empty(input.shape, dtype=input.dtype)
        else:
            output = cupy.empty(input.shape, dtype=types[-1])
    elif type(output) is type:
        if output not in types:
            raise RuntimeError('output type not supported')
        output = cupy.empty(input.shape, dtype=output)
    elif output.shape != input.shape:
        raise RuntimeError('output shape not correct')
    return output

def _reshape_nd(arr, ndim, axis):
    if False:
        return 10
    'Promote a 1d array to ndim with non-singleton size along axis.'
    nd_shape = (1,) * axis + (arr.size,) + (1,) * (ndim - axis - 1)
    return arr.reshape(nd_shape)

def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
    if False:
        i = 10
        return i + 15
    'Multidimensional Gaussian shift filter.\n\n    The array is multiplied with the Fourier transform of a (separable)\n    Gaussian kernel.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        sigma (float or sequence of float):  The sigma of the Gaussian kernel.\n            If a float, `sigma` is the same for all axes. If a sequence,\n            `sigma` has to contain one value for each axis.\n        n (int, optional):  If `n` is negative (default), then the input is\n            assumed to be the result of a complex fft. If `n` is larger than or\n            equal to zero, the input is assumed to be the result of a real fft,\n            and `n` gives the length of the array before transformation along\n            the real transform direction.\n        axis (int, optional): The axis of the real transform (only used when\n            ``n > -1``).\n        output (cupy.ndarray, optional):\n            If given, the result of shifting the input is placed in this array.\n\n    Returns:\n        output (cupy.ndarray): The filtered output.\n    '
    ndim = input.ndim
    output = _get_output_fourier(output, input)
    axis = internal._normalize_axis_index(axis, ndim)
    sigmas = _util._fix_sequence_arg(sigma, ndim, 'sigma')
    _core.elementwise_copy(input, output)
    for (ax, (sigmak, ax_size)) in enumerate(zip(sigmas, output.shape)):
        if ax == axis and n > 0:
            arr = cupy.arange(ax_size, dtype=output.real.dtype)
            arr /= n
        else:
            arr = cupy.fft.fftfreq(ax_size)
        arr = arr.astype(output.real.dtype, copy=False)
        arr *= arr
        scale = sigmak * sigmak / -2
        arr *= 4 * numpy.pi * numpy.pi * scale
        cupy.exp(arr, out=arr)
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr
    return output

def fourier_uniform(input, size, n=-1, axis=-1, output=None):
    if False:
        return 10
    'Multidimensional uniform shift filter.\n\n    The array is multiplied with the Fourier transform of a box of given size.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (float or sequence of float):  The sigma of the box used for\n            filtering. If a float, `size` is the same for all axes. If a\n            sequence, `size` has to contain one value for each axis.\n        n (int, optional):  If `n` is negative (default), then the input is\n            assumed to be the result of a complex fft. If `n` is larger than or\n            equal to zero, the input is assumed to be the result of a real fft,\n            and `n` gives the length of the array before transformation along\n            the real transform direction.\n        axis (int, optional): The axis of the real transform (only used when\n            ``n > -1``).\n        output (cupy.ndarray, optional):\n            If given, the result of shifting the input is placed in this array.\n\n    Returns:\n        output (cupy.ndarray): The filtered output.\n    '
    ndim = input.ndim
    output = _get_output_fourier(output, input)
    axis = internal._normalize_axis_index(axis, ndim)
    sizes = _util._fix_sequence_arg(size, ndim, 'size')
    _core.elementwise_copy(input, output)
    for (ax, (size, ax_size)) in enumerate(zip(sizes, output.shape)):
        if ax == axis and n > 0:
            arr = cupy.arange(ax_size, dtype=output.real.dtype)
            arr /= n
        else:
            arr = cupy.fft.fftfreq(ax_size)
        arr = arr.astype(output.real.dtype, copy=False)
        arr *= size
        cupy.sinc(arr, out=arr)
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr
    return output

def fourier_shift(input, shift, n=-1, axis=-1, output=None):
    if False:
        return 10
    'Multidimensional Fourier shift filter.\n\n    The array is multiplied with the Fourier transform of a shift operation.\n\n    Args:\n        input (cupy.ndarray): The input array. This should be in the Fourier\n            domain.\n        shift (float or sequence of float):  The size of shift. If a float,\n            `shift` is the same for all axes. If a sequence, `shift` has to\n            contain one value for each axis.\n        n (int, optional):  If `n` is negative (default), then the input is\n            assumed to be the result of a complex fft. If `n` is larger than or\n            equal to zero, the input is assumed to be the result of a real fft,\n            and `n` gives the length of the array before transformation along\n            the real transform direction.\n        axis (int, optional): The axis of the real transform (only used when\n            ``n > -1``).\n        output (cupy.ndarray, optional):\n            If given, the result of shifting the input is placed in this array.\n\n    Returns:\n        output (cupy.ndarray): The shifted output (in the Fourier domain).\n    '
    ndim = input.ndim
    output = _get_output_fourier(output, input, complex_only=True)
    axis = internal._normalize_axis_index(axis, ndim)
    shifts = _util._fix_sequence_arg(shift, ndim, 'shift')
    _core.elementwise_copy(input, output)
    for (ax, (shiftk, ax_size)) in enumerate(zip(shifts, output.shape)):
        if shiftk == 0:
            continue
        if ax == axis and n > 0:
            arr = cupy.arange(ax_size, dtype=output.dtype)
            arr *= -2j * numpy.pi * shiftk / n
        else:
            arr = cupy.fft.fftfreq(ax_size)
            arr = arr * (-2j * numpy.pi * shiftk)
        cupy.exp(arr, out=arr)
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr
    return output

def fourier_ellipsoid(input, size, n=-1, axis=-1, output=None):
    if False:
        i = 10
        return i + 15
    'Multidimensional ellipsoid Fourier filter.\n\n    The array is multiplied with the fourier transform of a ellipsoid of\n    given sizes.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (float or sequence of float):  The size of the box used for\n            filtering. If a float, `size` is the same for all axes. If a\n            sequence, `size` has to contain one value for each axis.\n        n (int, optional):  If `n` is negative (default), then the input is\n            assumed to be the result of a complex fft. If `n` is larger than or\n            equal to zero, the input is assumed to be the result of a real fft,\n            and `n` gives the length of the array before transformation along\n            the real transform direction.\n        axis (int, optional): The axis of the real transform (only used when\n            ``n > -1``).\n        output (cupy.ndarray, optional):\n            If given, the result of shifting the input is placed in this array.\n\n    Returns:\n        output (cupy.ndarray): The filtered output.\n    '
    ndim = input.ndim
    if ndim == 1:
        return fourier_uniform(input, size, n, axis, output)
    if ndim > 3:
        raise NotImplementedError('Only 1d, 2d and 3d inputs are supported')
    output = _get_output_fourier(output, input)
    axis = internal._normalize_axis_index(axis, ndim)
    sizes = _util._fix_sequence_arg(size, ndim, 'size')
    _core.elementwise_copy(input, output)
    distance = 0
    for (ax, (size, ax_size)) in enumerate(zip(sizes, output.shape)):
        if ax == axis and n > 0:
            arr = cupy.arange(ax_size, dtype=output.real.dtype)
            arr *= numpy.pi * size / n
        else:
            arr = cupy.fft.fftfreq(ax_size)
            arr *= numpy.pi * size
        arr = arr.astype(output.real.dtype, copy=False)
        arr *= arr
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        distance = distance + arr
    cupy.sqrt(distance, out=distance)
    if ndim == 2:
        special.j1(distance, out=output)
        output *= 2
        output /= distance
    elif ndim == 3:
        cupy.sin(distance, out=output)
        output -= distance * cupy.cos(distance)
        output *= 3
        output /= distance ** 3
    output[(0,) * ndim] = 1.0
    output *= input
    return output