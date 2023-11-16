import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic

def correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        while True:
            i = 10
    "Multi-dimensional correlate.\n\n    The array is correlated with the given kernel.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        weights (cupy.ndarray): Array of weights, same number of dimensions as\n            input\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of correlate.\n\n    .. seealso:: :func:`scipy.ndimage.correlate`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    return _correlate_or_convolve(input, weights, output, mode, cval, origin)

def convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    "Multi-dimensional convolution.\n\n    The array is convolved with the given kernel.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        weights (cupy.ndarray): Array of weights, same number of dimensions as\n            input\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of convolution.\n\n    .. seealso:: :func:`scipy.ndimage.convolve`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    return _correlate_or_convolve(input, weights, output, mode, cval, origin, True)

def correlate1d(input, weights, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        i = 10
        return i + 15
    "One-dimensional correlate.\n\n    The array is correlated with the given kernel.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        weights (cupy.ndarray): One-dimensional array of weights\n        axis (int): The axis of input along which to calculate. Default is -1.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int): The origin parameter controls the placement of the\n            filter, relative to the center of the current element of the\n            input. Default is ``0``.\n\n    Returns:\n        cupy.ndarray: The result of the 1D correlation.\n\n    .. seealso:: :func:`scipy.ndimage.correlate1d`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    (weights, origins) = _filters_core._convert_1d_args(input.ndim, weights, origin, axis)
    return _correlate_or_convolve(input, weights, output, mode, cval, origins)

def convolve1d(input, weights, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        i = 10
        return i + 15
    "One-dimensional convolution.\n\n    The array is convolved with the given kernel.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        weights (cupy.ndarray): One-dimensional array of weights\n        axis (int): The axis of input along which to calculate. Default is -1.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int): The origin parameter controls the placement of the\n            filter, relative to the center of the current element of the\n            input. Default is ``0``.\n    Returns:\n        cupy.ndarray: The result of the 1D convolution.\n\n    .. seealso:: :func:`scipy.ndimage.convolve1d`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    (weights, origins) = _filters_core._convert_1d_args(input.ndim, weights, origin, axis)
    return _correlate_or_convolve(input, weights, output, mode, cval, origins, True)

def _correlate_or_convolve(input, weights, output, mode, cval, origin, convolution=False):
    if False:
        while True:
            i = 10
    (origins, int_type) = _filters_core._check_nd_args(input, weights, mode, origin)
    if weights.size == 0:
        return cupy.zeros_like(input)
    _util._check_cval(mode, cval, _util._is_integer_output(output, input))
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        origins = list(origins)
        for (i, wsize) in enumerate(weights.shape):
            origins[i] = -origins[i]
            if wsize % 2 == 0:
                origins[i] -= 1
        origins = tuple(origins)
    elif weights.dtype.kind == 'c':
        weights = weights.conj()
    weights_dtype = _util._get_weights_dtype(input, weights)
    offsets = _filters_core._origins_to_offsets(origins, weights.shape)
    kernel = _get_correlate_kernel(mode, weights.shape, int_type, offsets, cval)
    output = _filters_core._call_kernel(kernel, input, weights, output, weights_dtype=weights_dtype)
    return output

@cupy._util.memoize(for_each_device=True)
def _get_correlate_kernel(mode, w_shape, int_type, offsets, cval):
    if False:
        for i in range(10):
            print('nop')
    return _filters_core._generate_nd_kernel('correlate', 'W sum = (W)0;', 'sum += cast<W>({value}) * wval;', 'y = cast<Y>(sum);', mode, w_shape, int_type, offsets, cval, ctype='W')

def _run_1d_correlates(input, params, get_weights, output, mode, cval, origin=0):
    if False:
        return 10
    '\n    Enhanced version of _run_1d_filters that uses correlate1d as the filter\n    function. The params are a list of values to pass to the get_weights\n    callable given. If duplicate param values are found, the weights are\n    reused from the first invocation of get_weights. The get_weights callable\n    must return a 1D array of weights to give to correlate1d.\n    '
    wghts = {}
    for param in params:
        if param not in wghts:
            wghts[param] = get_weights(param)
    wghts = [wghts[param] for param in params]
    return _filters_core._run_1d_filters([None if w is None else correlate1d for w in wghts], input, wghts, output, mode, cval, origin)

def uniform_filter1d(input, size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        i = 10
        return i + 15
    "One-dimensional uniform filter along the given axis.\n\n    The lines of the array along the given axis are filtered with a uniform\n    filter of the given size.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (int): Length of the uniform filter.\n        axis (int): The axis of input along which to calculate. Default is -1.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int): The origin parameter controls the placement of the\n            filter, relative to the center of the current element of the\n            input. Default is ``0``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.uniform_filter1d`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    weights_dtype = _util._init_weights_dtype(input)
    weights = cupy.full(size, 1 / size, dtype=weights_dtype)
    return correlate1d(input, weights, axis, output, mode, cval, origin)

def uniform_filter(input, size=3, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        return 10
    "Multi-dimensional uniform filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (int or sequence of int): Lengths of the uniform filter for each\n            dimension. A single value applies to all axes.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int or sequence of int): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of ``0`` is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.uniform_filter`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    sizes = _util._fix_sequence_arg(size, input.ndim, 'size', int)
    weights_dtype = _util._init_weights_dtype(input)

    def get(size, dtype=weights_dtype):
        if False:
            i = 10
            return i + 15
        return None if size <= 1 else cupy.full(size, 1 / size, dtype=dtype)
    return _run_1d_correlates(input, sizes, get, output, mode, cval, origin)

def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
    if False:
        return 10
    "One-dimensional Gaussian filter along the given axis.\n\n    The lines of the array along the given axis are filtered with a Gaussian\n    filter of the given standard deviation.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        sigma (scalar): Standard deviation for Gaussian kernel.\n        axis (int): The axis of input along which to calculate. Default is -1.\n        order (int): An order of ``0``, the default, corresponds to convolution\n            with a Gaussian kernel. A positive order corresponds to convolution\n            with that derivative of a Gaussian.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        truncate (float): Truncate the filter at this many standard deviations.\n            Default is ``4.0``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.gaussian_filter1d`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    radius = int(float(truncate) * float(sigma) + 0.5)
    weights_dtype = _util._init_weights_dtype(input)
    weights = _gaussian_kernel1d(sigma, int(order), radius, dtype=weights_dtype)
    return correlate1d(input, weights, axis, output, mode, cval)

def gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
    if False:
        while True:
            i = 10
    "Multi-dimensional Gaussian filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        sigma (scalar or sequence of scalar): Standard deviations for each axis\n            of Gaussian kernel. A single value applies to all axes.\n        order (int or sequence of scalar): An order of ``0``, the default,\n            corresponds to convolution with a Gaussian kernel. A positive order\n            corresponds to convolution with that derivative of a Gaussian. A\n            single value applies to all axes.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        truncate (float): Truncate the filter at this many standard deviations.\n            Default is ``4.0``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.gaussian_filter`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    sigmas = _util._fix_sequence_arg(sigma, input.ndim, 'sigma', float)
    orders = _util._fix_sequence_arg(order, input.ndim, 'order', int)
    truncate = float(truncate)
    weights_dtype = _util._init_weights_dtype(input)

    def get(param):
        if False:
            print('Hello World!')
        (sigma, order) = param
        radius = int(truncate * float(sigma) + 0.5)
        if radius <= 0:
            return None
        return _gaussian_kernel1d(sigma, order, radius, dtype=weights_dtype)
    return _run_1d_correlates(input, list(zip(sigmas, orders)), get, output, mode, cval, 0)

def _gaussian_kernel1d(sigma, order, radius, dtype=cupy.float64):
    if False:
        i = 10
        return i + 15
    '\n    Computes a 1-D Gaussian correlation kernel.\n    '
    if order < 0:
        raise ValueError('order must be non-negative')
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius + 1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x /= phi_x.sum()
    if order == 0:
        return cupy.asarray(phi_x)
    exponent_range = numpy.arange(order + 1)
    q = numpy.zeros(order + 1)
    q[0] = 1
    D = numpy.diag(exponent_range[1:], 1)
    P = numpy.diag(numpy.ones(order) / -sigma2, -1)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return cupy.asarray((q * phi_x)[::-1], dtype=dtype)

def prewitt(input, axis=-1, output=None, mode='reflect', cval=0.0):
    if False:
        print('Hello World!')
    "Compute a Prewitt filter along the given axis.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        axis (int): The axis of input along which to calculate. Default is -1.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.prewitt`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    weights_dtype = _util._init_weights_dtype(input)
    weights = cupy.ones(3, dtype=weights_dtype)
    return _prewitt_or_sobel(input, axis, output, mode, cval, weights)

def sobel(input, axis=-1, output=None, mode='reflect', cval=0.0):
    if False:
        while True:
            i = 10
    "Compute a Sobel filter along the given axis.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        axis (int): The axis of input along which to calculate. Default is -1.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.sobel`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    weights_dtype = _util._init_weights_dtype(input)
    return _prewitt_or_sobel(input, axis, output, mode, cval, cupy.array([1, 2, 1], dtype=weights_dtype))

def _prewitt_or_sobel(input, axis, output, mode, cval, weights):
    if False:
        return 10
    axis = internal._normalize_axis_index(axis, input.ndim)

    def get(is_diff):
        if False:
            for i in range(10):
                print('nop')
        return cupy.array([-1, 0, 1], dtype=weights.dtype) if is_diff else weights
    return _run_1d_correlates(input, [a == axis for a in range(input.ndim)], get, output, mode, cval)

def generic_laplace(input, derivative2, output=None, mode='reflect', cval=0.0, extra_arguments=(), extra_keywords=None):
    if False:
        print('Hello World!')
    "Multi-dimensional Laplace filter using a provided second derivative\n    function.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        derivative2 (callable): Function or other callable with the following\n            signature that is called once per axis::\n\n                derivative2(input, axis, output, mode, cval,\n                            *extra_arguments, **extra_keywords)\n\n            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an\n            ``int`` from ``0`` to the number of dimensions, and ``mode``,\n            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values\n            given to this function.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        extra_arguments (sequence, optional):\n            Sequence of extra positional arguments to pass to ``derivative2``.\n        extra_keywords (dict, optional):\n            dict of extra keyword arguments to pass ``derivative2``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.generic_laplace`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    if extra_keywords is None:
        extra_keywords = {}
    ndim = input.ndim
    modes = _util._fix_sequence_arg(mode, ndim, 'mode', _util._check_mode)
    output = _util._get_output(output, input)
    if ndim == 0:
        _core.elementwise_copy(input, output)
        return output
    derivative2(input, 0, output, modes[0], cval, *extra_arguments, **extra_keywords)
    if ndim > 1:
        tmp = _util._get_output(output.dtype, input)
        for i in range(1, ndim):
            derivative2(input, i, tmp, modes[i], cval, *extra_arguments, **extra_keywords)
            output += tmp
    return output

def laplace(input, output=None, mode='reflect', cval=0.0):
    if False:
        i = 10
        return i + 15
    "Multi-dimensional Laplace filter based on approximate second\n    derivatives.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.laplace`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    weights_dtype = _util._init_weights_dtype(input)
    weights = cupy.array([1, -2, 1], dtype=weights_dtype)

    def derivative2(input, axis, output, mode, cval):
        if False:
            while True:
                i = 10
        return correlate1d(input, weights, axis, output, mode, cval)
    return generic_laplace(input, derivative2, output, mode, cval)

def gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs):
    if False:
        i = 10
        return i + 15
    "Multi-dimensional Laplace filter using Gaussian second derivatives.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        sigma (scalar or sequence of scalar): Standard deviations for each axis\n            of Gaussian kernel. A single value applies to all axes.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        kwargs (dict, optional):\n            dict of extra keyword arguments to pass ``gaussian_filter()``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.gaussian_laplace`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "

    def derivative2(input, axis, output, mode, cval):
        if False:
            return 10
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(input, sigma, order, output, mode, cval, **kwargs)
    return generic_laplace(input, derivative2, output, mode, cval)

def generic_gradient_magnitude(input, derivative, output=None, mode='reflect', cval=0.0, extra_arguments=(), extra_keywords=None):
    if False:
        for i in range(10):
            print('nop')
    "Multi-dimensional gradient magnitude filter using a provided derivative\n    function.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        derivative (callable): Function or other callable with the following\n            signature that is called once per axis::\n\n                derivative(input, axis, output, mode, cval,\n                           *extra_arguments, **extra_keywords)\n\n            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an\n            ``int`` from ``0`` to the number of dimensions, and ``mode``,\n            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values\n            given to this function.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        extra_arguments (sequence, optional):\n            Sequence of extra positional arguments to pass to ``derivative2``.\n        extra_keywords (dict, optional):\n            dict of extra keyword arguments to pass ``derivative2``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.generic_gradient_magnitude`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "
    if extra_keywords is None:
        extra_keywords = {}
    ndim = input.ndim
    modes = _util._fix_sequence_arg(mode, ndim, 'mode', _util._check_mode)
    output = _util._get_output(output, input)
    if ndim == 0:
        _core.elementwise_copy(input, output)
        return output
    derivative(input, 0, output, modes[0], cval, *extra_arguments, **extra_keywords)
    output *= output
    if ndim > 1:
        tmp = _util._get_output(output.dtype, input)
        for i in range(1, ndim):
            derivative(input, i, tmp, modes[i], cval, *extra_arguments, **extra_keywords)
            tmp *= tmp
            output += tmp
    return cupy.sqrt(output, output, casting='unsafe')

def gaussian_gradient_magnitude(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs):
    if False:
        print('Hello World!')
    "Multi-dimensional gradient magnitude using Gaussian derivatives.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        sigma (scalar or sequence of scalar): Standard deviations for each axis\n            of Gaussian kernel. A single value applies to all axes.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        kwargs (dict, optional):\n            dict of extra keyword arguments to pass ``gaussian_filter()``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.gaussian_gradient_magnitude`\n\n    .. note::\n        When the output data type is integral (or when no output is provided\n        and input is integral) the results may not perfectly match the results\n        from SciPy due to floating-point rounding of intermediate results.\n    "

    def derivative(input, axis, output, mode, cval):
        if False:
            while True:
                i = 10
        order = [0] * input.ndim
        order[axis] = 1
        return gaussian_filter(input, sigma, order, output, mode, cval, **kwargs)
    return generic_gradient_magnitude(input, derivative, output, mode, cval)

def minimum_filter(input, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    "Multi-dimensional minimum filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (int or sequence of int): One of ``size`` or ``footprint`` must be\n            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise\n            ``footprint = cupy.ones(size)`` with ``size`` automatically made to\n            match the number of dimensions in ``input``.\n        footprint (cupy.ndarray): a boolean array which specifies which of the\n            elements within this shape will get passed to the filter function.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int or sequence of int): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.minimum_filter`\n    "
    return _min_or_max_filter(input, size, footprint, None, output, mode, cval, origin, 'min')

def maximum_filter(input, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        while True:
            i = 10
    "Multi-dimensional maximum filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (int or sequence of int): One of ``size`` or ``footprint`` must be\n            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise\n            ``footprint = cupy.ones(size)`` with ``size`` automatically made to\n            match the number of dimensions in ``input``.\n        footprint (cupy.ndarray): a boolean array which specifies which of the\n            elements within this shape will get passed to the filter function.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int or sequence of int): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.maximum_filter`\n    "
    return _min_or_max_filter(input, size, footprint, None, output, mode, cval, origin, 'max')

def _min_or_max_filter(input, size, ftprnt, structure, output, mode, cval, origin, func):
    if False:
        print('Hello World!')
    (sizes, ftprnt, structure) = _filters_core._check_size_footprint_structure(input.ndim, size, ftprnt, structure)
    if cval is cupy.nan:
        raise NotImplementedError('NaN cval is unsupported')
    if sizes is not None:
        fltr = minimum_filter1d if func == 'min' else maximum_filter1d
        return _filters_core._run_1d_filters([fltr if size > 1 else None for size in sizes], input, sizes, output, mode, cval, origin)
    (origins, int_type) = _filters_core._check_nd_args(input, ftprnt, mode, origin, 'footprint')
    if structure is not None and structure.ndim != input.ndim:
        raise RuntimeError('structure array has incorrect shape')
    if ftprnt.size == 0:
        return cupy.zeros_like(input)
    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func, offsets, float(cval), int_type, has_structure=structure is not None, has_central_value=bool(ftprnt[offsets]))
    return _filters_core._call_kernel(kernel, input, ftprnt, output, structure, weights_dtype=bool)

def minimum_filter1d(input, size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    "Compute the minimum filter along a single axis.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (int): Length of the minimum filter.\n        axis (int): The axis of input along which to calculate. Default is -1.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int): The origin parameter controls the placement of the\n            filter, relative to the center of the current element of the\n            input. Default is ``0``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.minimum_filter1d`\n    "
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, 'min')

def maximum_filter1d(input, size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        i = 10
        return i + 15
    "Compute the maximum filter along a single axis.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (int): Length of the maximum filter.\n        axis (int): The axis of input along which to calculate. Default is -1.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int): The origin parameter controls the placement of the\n            filter, relative to the center of the current element of the\n            input. Default is ``0``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.maximum_filter1d`\n    "
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, 'max')

def _min_or_max_1d(input, size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0, func='min'):
    if False:
        while True:
            i = 10
    ftprnt = cupy.ones(size, dtype=bool)
    (ftprnt, origin) = _filters_core._convert_1d_args(input.ndim, ftprnt, origin, axis)
    (origins, int_type) = _filters_core._check_nd_args(input, ftprnt, mode, origin, 'footprint')
    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func, offsets, float(cval), int_type, has_weights=False)
    return _filters_core._call_kernel(kernel, input, None, output, weights_dtype=bool)

@cupy._util.memoize(for_each_device=True)
def _get_min_or_max_kernel(mode, w_shape, func, offsets, cval, int_type, has_weights=True, has_structure=False, has_central_value=True):
    if False:
        while True:
            i = 10
    ctype = 'X' if has_weights else 'double'
    value = '{value}'
    if not has_weights:
        value = 'cast<double>({})'.format(value)
    if has_structure:
        value += ('-' if func == 'min' else '+') + 'cast<X>(sval)'
    if has_central_value:
        pre = '{} value = x[i];'
        found = 'value = {func}({value}, value);'
    else:
        pre = '{} value; bool set = false;'
        found = 'value = set ? {func}({value}, value) : {value}; set=true;'
    return _filters_core._generate_nd_kernel(func, pre.format(ctype), found.format(func=func, value=value), 'y = cast<Y>(value);', mode, w_shape, int_type, offsets, cval, ctype=ctype, has_weights=has_weights, has_structure=has_structure)

def rank_filter(input, rank, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        while True:
            i = 10
    "Multi-dimensional rank filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        rank (int): The rank of the element to get. Can be negative to count\n            from the largest value, e.g. ``-1`` indicates the largest value.\n        size (int or sequence of int): One of ``size`` or ``footprint`` must be\n            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise\n            ``footprint = cupy.ones(size)`` with ``size`` automatically made to\n            match the number of dimensions in ``input``.\n        footprint (cupy.ndarray): a boolean array which specifies which of the\n            elements within this shape will get passed to the filter function.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int or sequence of int): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.rank_filter`\n    "
    rank = int(rank)
    return _rank_filter(input, lambda fs: rank + fs if rank < 0 else rank, size, footprint, output, mode, cval, origin)

def median_filter(input, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        i = 10
        return i + 15
    "Multi-dimensional median filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (int or sequence of int): One of ``size`` or ``footprint`` must be\n            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise\n            ``footprint = cupy.ones(size)`` with ``size`` automatically made to\n            match the number of dimensions in ``input``.\n        footprint (cupy.ndarray): a boolean array which specifies which of the\n            elements within this shape will get passed to the filter function.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int or sequence of int): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.median_filter`\n    "
    return _rank_filter(input, lambda fs: fs // 2, size, footprint, output, mode, cval, origin)

def percentile_filter(input, percentile, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        for i in range(10):
            print('nop')
    "Multi-dimensional percentile filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        percentile (scalar): The percentile of the element to get (from ``0``\n            to ``100``). Can be negative, thus ``-20`` equals ``80``.\n        size (int or sequence of int): One of ``size`` or ``footprint`` must be\n            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise\n            ``footprint = cupy.ones(size)`` with ``size`` automatically made to\n            match the number of dimensions in ``input``.\n        footprint (cupy.ndarray): a boolean array which specifies which of the\n            elements within this shape will get passed to the filter function.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int or sequence of int): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. seealso:: :func:`scipy.ndimage.percentile_filter`\n    "
    percentile = float(percentile)
    if percentile < 0.0:
        percentile += 100.0
    if percentile < 0.0 or percentile > 100.0:
        raise RuntimeError('invalid percentile')
    if percentile == 100.0:

        def get_rank(fs):
            if False:
                i = 10
                return i + 15
            return fs - 1
    else:

        def get_rank(fs):
            if False:
                for i in range(10):
                    print('nop')
            return int(float(fs) * percentile / 100.0)
    return _rank_filter(input, get_rank, size, footprint, output, mode, cval, origin)

def _rank_filter(input, get_rank, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    (_, footprint, _) = _filters_core._check_size_footprint_structure(input.ndim, size, footprint, None, force_footprint=True)
    if cval is cupy.nan:
        raise NotImplementedError('NaN cval is unsupported')
    (origins, int_type) = _filters_core._check_nd_args(input, footprint, mode, origin, 'footprint')
    if footprint.size == 0:
        return cupy.zeros_like(input)
    filter_size = int(footprint.sum())
    rank = get_rank(filter_size)
    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    if rank == 0:
        return _min_or_max_filter(input, None, footprint, None, output, mode, cval, origins, 'min')
    if rank == filter_size - 1:
        return _min_or_max_filter(input, None, footprint, None, output, mode, cval, origins, 'max')
    offsets = _filters_core._origins_to_offsets(origins, footprint.shape)
    kernel = _get_rank_kernel(filter_size, rank, mode, footprint.shape, offsets, float(cval), int_type)
    return _filters_core._call_kernel(kernel, input, footprint, output, weights_dtype=bool)
__SHELL_SORT = '\n__device__ void sort(X *array, int size) {{\n    int gap = {gap};\n    while (gap > 1) {{\n        gap /= 3;\n        for (int i = gap; i < size; ++i) {{\n            X value = array[i];\n            int j = i - gap;\n            while (j >= 0 && value < array[j]) {{\n                array[j + gap] = array[j];\n                j -= gap;\n            }}\n            array[j + gap] = value;\n        }}\n    }}\n}}'

@cupy._util.memoize()
def _get_shell_gap(filter_size):
    if False:
        print('Hello World!')
    gap = 1
    while gap < filter_size:
        gap = 3 * gap + 1
    return gap

@cupy._util.memoize(for_each_device=True)
def _get_rank_kernel(filter_size, rank, mode, w_shape, offsets, cval, int_type):
    if False:
        return 10
    s_rank = min(rank, filter_size - rank - 1)
    if s_rank <= 80:
        if s_rank == rank:
            comp_op = '<'
        else:
            comp_op = '>'
        array_size = s_rank + 2
        found_post = '\n            if (iv > {rank} + 1) {{{{\n                int target_iv = 0;\n                X target_val = values[0];\n                for (int jv = 1; jv <= {rank} + 1; jv++) {{{{\n                    if (target_val {comp_op} values[jv]) {{{{\n                        target_val = values[jv];\n                        target_iv = jv;\n                    }}}}\n                }}}}\n                if (target_iv <= {rank}) {{{{\n                    values[target_iv] = values[{rank} + 1];\n                }}}}\n                iv = {rank} + 1;\n            }}}}'.format(rank=s_rank, comp_op=comp_op)
        post = '\n            X target_val = values[0];\n            for (int jv = 1; jv <= {rank}; jv++) {{\n                if (target_val {comp_op} values[jv]) {{\n                    target_val = values[jv];\n                }}\n            }}\n            y=cast<Y>(target_val);'.format(rank=s_rank, comp_op=comp_op)
        sorter = ''
    else:
        array_size = filter_size
        found_post = ''
        post = 'sort(values,{});\ny=cast<Y>(values[{}]);'.format(filter_size, rank)
        sorter = __SHELL_SORT.format(gap=_get_shell_gap(filter_size))
    return _filters_core._generate_nd_kernel('rank_{}_{}'.format(filter_size, rank), 'int iv = 0;\nX values[{}];'.format(array_size), 'values[iv++] = {value};' + found_post, post, mode, w_shape, int_type, offsets, cval, preamble=sorter)

def generic_filter(input, function, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    "Compute a multi-dimensional filter using the provided raw kernel or\n    reduction kernel.\n\n    Unlike the scipy.ndimage function, this does not support the\n    ``extra_arguments`` or ``extra_keywordsdict`` arguments and has significant\n    restrictions on the ``function`` provided.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        function (cupy.ReductionKernel or cupy.RawKernel):\n            The kernel or function to apply to each region.\n        size (int or sequence of int): One of ``size`` or ``footprint`` must be\n            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise\n            ``footprint = cupy.ones(size)`` with ``size`` automatically made to\n            match the number of dimensions in ``input``.\n        footprint (cupy.ndarray): a boolean array which specifies which of the\n            elements within this shape will get passed to the filter function.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. note::\n        If the `function` is a :class:`cupy.RawKernel` then it must be for a\n        function that has the following signature. Unlike most functions, this\n        should not utilize `blockDim`/`blockIdx`/`threadIdx`::\n\n            __global__ void func(double *buffer, int filter_size,\n                                 double *return_value)\n\n        If the `function` is a :class:`cupy.ReductionKernel` then it must be\n        for a kernel that takes 1 array input and produces 1 'scalar' output.\n\n    .. seealso:: :func:`scipy.ndimage.generic_filter`\n    "
    (_, footprint, _) = _filters_core._check_size_footprint_structure(input.ndim, size, footprint, None, 2, True)
    filter_size = int(footprint.sum())
    (origins, int_type) = _filters_core._check_nd_args(input, footprint, mode, origin, 'footprint')
    in_dtype = input.dtype
    sub = _filters_generic._get_sub_kernel(function)
    if footprint.size == 0:
        return cupy.zeros_like(input)
    output = _util._get_output(output, input)
    offsets = _filters_core._origins_to_offsets(origins, footprint.shape)
    args = (filter_size, mode, footprint.shape, offsets, float(cval), int_type)
    if isinstance(sub, cupy.RawKernel):
        kernel = _filters_generic._get_generic_filter_raw(sub, *args)
    elif isinstance(sub, cupy.ReductionKernel):
        kernel = _filters_generic._get_generic_filter_red(sub, in_dtype, output.dtype, *args)
    return _filters_core._call_kernel(kernel, input, footprint, output, weights_dtype=bool)

def generic_filter1d(input, function, filter_size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        while True:
            i = 10
    "Compute a 1D filter along the given axis using the provided raw kernel.\n\n    Unlike the scipy.ndimage function, this does not support the\n    ``extra_arguments`` or ``extra_keywordsdict`` arguments and has significant\n    restrictions on the ``function`` provided.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        function (cupy.RawKernel): The kernel to apply along each axis.\n        filter_size (int): Length of the filter.\n        axis (int): The axis of input along which to calculate. Default is -1.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output. Default is is same dtype as the input.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``'constant'``. Default is ``0.0``.\n        origin (int): The origin parameter controls the placement of the\n            filter, relative to the center of the current element of the\n            input. Default is ``0``.\n\n    Returns:\n        cupy.ndarray: The result of the filtering.\n\n    .. note::\n        The provided function (as a RawKernel) must have the following\n        signature. Unlike most functions, this should not utilize\n        `blockDim`/`blockIdx`/`threadIdx`::\n\n            __global__ void func(double *input_line, ptrdiff_t input_length,\n                                 double *output_line, ptrdiff_t output_length)\n\n    .. seealso:: :func:`scipy.ndimage.generic_filter1d`\n    "
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    if not isinstance(function, cupy.RawKernel):
        raise TypeError('bad function type')
    if filter_size < 1:
        raise RuntimeError('invalid filter size')
    axis = internal._normalize_axis_index(axis, input.ndim)
    origin = _util._check_origin(origin, filter_size)
    _util._check_mode(mode)
    output = _util._get_output(output, input)
    in_ctype = cupy._core._scalar.get_typename(input.dtype)
    out_ctype = cupy._core._scalar.get_typename(output.dtype)
    int_type = _util._get_inttype(input)
    n_lines = input.size // input.shape[axis]
    kernel = _filters_generic._get_generic_filter1d(function, input.shape[axis], n_lines, filter_size, origin, mode, float(cval), in_ctype, out_ctype, int_type)
    data = cupy.array((axis, input.ndim) + input.shape + input.strides + output.strides, dtype=cupy.int32 if int_type == 'int' else cupy.int64)
    kernel(((n_lines + 128 - 1) // 128,), (128,), (input, output, data))
    return output