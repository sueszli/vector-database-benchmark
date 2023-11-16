import warnings
from functools import partial
import numpy as np
from astropy import units as u
from astropy.modeling.convolution import Convolution
from astropy.modeling.core import SPECIAL_OPERATORS, CompoundModel
from astropy.nddata import support_nddata
from astropy.utils.console import human_file_size
from astropy.utils.exceptions import AstropyUserWarning
from ._convolve import _convolveNd_c
from .core import MAX_NORMALIZATION, Kernel, Kernel1D, Kernel2D
from .utils import KernelSizeError, has_even_axis
_good_sizes = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000])
_good_range = int(np.log10(_good_sizes[-1]))
__doctest_requires__ = {('convolve_fft',): ['scipy.fft']}
BOUNDARY_OPTIONS = [None, 'fill', 'wrap', 'extend']

def _next_fast_lengths(shape):
    if False:
        while True:
            i = 10
    '\n    Find optimal or good sizes to pad an array of ``shape`` to for better\n    performance with `numpy.fft.*fft` and `scipy.fft.*fft`.\n    Calculated directly with `scipy.fft.next_fast_len`, if available; otherwise\n    looked up from list and scaled by powers of 10, if necessary.\n    '
    try:
        import scipy.fft
        return np.array([scipy.fft.next_fast_len(j) for j in shape])
    except ImportError:
        pass
    newshape = np.empty(len(np.atleast_1d(shape)), dtype=int)
    for (i, j) in enumerate(shape):
        scale = 10 ** max(int(np.ceil(np.log10(j))) - _good_range, 0)
        for n in _good_sizes:
            if n * scale >= j:
                newshape[i] = n * scale
                break
        else:
            raise ValueError(f'No next fast length for {j} found in list of _good_sizes <= {_good_sizes[-1] * scale}.')
    return newshape

def _copy_input_if_needed(input, dtype=float, order='C', nan_treatment=None, mask=None, fill_value=None):
    if False:
        print('Hello World!')
    input = input.array if isinstance(input, Kernel) else input
    if hasattr(input, 'unit'):
        input = input.value
    output = input
    try:
        if nan_treatment == 'fill' or np.ma.is_masked(input) or mask is not None:
            if np.ma.is_masked(input):
                output = np.array(input, dtype=dtype, copy=False, order=order, subok=True)
                output = output.filled(fill_value)
            else:
                output = np.array(input, dtype=dtype, copy=True, order=order, subok=False)
            if mask is not None:
                output[mask != 0] = fill_value
        else:
            output = np.array(input, dtype=dtype, copy=False, order=order, subok=True)
    except (TypeError, ValueError) as e:
        raise TypeError('input should be a Numpy array or something convertible into a float array', e)
    return output

@support_nddata(data='array')
def convolve(array, kernel, boundary='fill', fill_value=0.0, nan_treatment='interpolate', normalize_kernel=True, mask=None, preserve_nan=False, normalization_zero_tol=1e-08):
    if False:
        return 10
    '\n    Convolve an array with a kernel.\n\n    This routine differs from `scipy.ndimage.convolve` because\n    it includes a special treatment for ``NaN`` values. Rather than\n    including ``NaN`` values in the array in the convolution calculation, which\n    causes large ``NaN`` holes in the convolved array, ``NaN`` values are\n    replaced with interpolated values using the kernel as an interpolation\n    function.\n\n    Parameters\n    ----------\n    array : `~astropy.nddata.NDData` or array-like\n        The array to convolve. This should be a 1, 2, or 3-dimensional array\n        or a list or a set of nested lists representing a 1, 2, or\n        3-dimensional array.  If an `~astropy.nddata.NDData`, the ``mask`` of\n        the `~astropy.nddata.NDData` will be used as the ``mask`` argument.\n    kernel : `numpy.ndarray` or `~astropy.convolution.Kernel`\n        The convolution kernel. The number of dimensions should match those for\n        the array, and the dimensions should be odd in all directions.  If a\n        masked array, the masked values will be replaced by ``fill_value``.\n    boundary : str, optional\n        A flag indicating how to handle boundaries:\n            * `None`\n                Set the ``result`` values to zero where the kernel\n                extends beyond the edge of the array.\n            * \'fill\'\n                Set values outside the array boundary to ``fill_value`` (default).\n            * \'wrap\'\n                Periodic boundary that wrap to the other side of ``array``.\n            * \'extend\'\n                Set values outside the array to the nearest ``array``\n                value.\n    fill_value : float, optional\n        The value to use outside the array when using ``boundary=\'fill\'``.\n    normalize_kernel : bool, optional\n        Whether to normalize the kernel to have a sum of one.\n    nan_treatment : {\'interpolate\', \'fill\'}, optional\n        The method used to handle NaNs in the input ``array``:\n            * ``\'interpolate\'``: ``NaN`` values are replaced with\n              interpolated values using the kernel as an interpolation\n              function. Note that if the kernel has a sum equal to\n              zero, NaN interpolation is not possible and will raise an\n              exception.\n            * ``\'fill\'``: ``NaN`` values are replaced by ``fill_value``\n              prior to convolution.\n    preserve_nan : bool, optional\n        After performing convolution, should pixels that were originally NaN\n        again become NaN?\n    mask : None or ndarray, optional\n        A "mask" array.  Shape must match ``array``, and anything that is masked\n        (i.e., not 0/`False`) will be set to NaN for the convolution.  If\n        `None`, no masking will be performed unless ``array`` is a masked array.\n        If ``mask`` is not `None` *and* ``array`` is a masked array, a pixel is\n        masked if it is masked in either ``mask`` *or* ``array.mask``.\n    normalization_zero_tol : float, optional\n        The absolute tolerance on whether the kernel is different than zero.\n        If the kernel sums to zero to within this precision, it cannot be\n        normalized. Default is "1e-8".\n\n    Returns\n    -------\n    result : `numpy.ndarray`\n        An array with the same dimensions and as the input array,\n        convolved with kernel.  The data type depends on the input\n        array type.  If array is a floating point type, then the\n        return array keeps the same data type, otherwise the type\n        is ``numpy.float``.\n\n    Notes\n    -----\n    For masked arrays, masked values are treated as NaNs.  The convolution\n    is always done at ``numpy.float`` precision.\n    '
    if boundary not in BOUNDARY_OPTIONS:
        raise ValueError(f'Invalid boundary option: must be one of {BOUNDARY_OPTIONS}')
    if nan_treatment not in ('interpolate', 'fill'):
        raise ValueError("nan_treatment must be one of 'interpolate','fill'")
    n_threads = 1
    passed_kernel = kernel
    passed_array = array
    array_internal = _copy_input_if_needed(passed_array, dtype=float, order='C', nan_treatment=nan_treatment, mask=mask, fill_value=np.nan)
    array_dtype = getattr(passed_array, 'dtype', array_internal.dtype)
    kernel_internal = _copy_input_if_needed(passed_kernel, dtype=float, order='C', nan_treatment=None, mask=None, fill_value=fill_value)
    if has_even_axis(kernel_internal):
        raise KernelSizeError('Kernel size must be odd in all axes.')
    if isinstance(passed_array, Kernel) and isinstance(passed_kernel, Kernel):
        warnings.warn("Both array and kernel are Kernel instances, hardwiring the following parameters: boundary='fill', fill_value=0, normalize_Kernel=True, nan_treatment='interpolate'", AstropyUserWarning)
        boundary = 'fill'
        fill_value = 0
        normalize_kernel = True
        nan_treatment = 'interpolate'
    if array_internal.ndim == 0:
        raise Exception('cannot convolve 0-dimensional arrays')
    elif array_internal.ndim > 3:
        raise NotImplementedError('convolve only supports 1, 2, and 3-dimensional arrays at this time')
    elif array_internal.ndim != kernel_internal.ndim:
        raise Exception('array and kernel have differing number of dimensions.')
    array_shape = np.array(array_internal.shape)
    kernel_shape = np.array(kernel_internal.shape)
    pad_width = kernel_shape // 2
    if boundary is None and (not np.all(array_shape > 2 * pad_width)):
        raise KernelSizeError("for boundary=None all kernel axes must be smaller than array's - use boundary in ['fill', 'extend', 'wrap'] instead.")
    nan_interpolate = nan_treatment == 'interpolate' and np.isnan(array_internal.sum())
    if normalize_kernel or nan_interpolate:
        kernel_sum = kernel_internal.sum()
        kernel_sums_to_zero = np.isclose(kernel_sum, 0, atol=normalization_zero_tol)
        if kernel_sum < 1.0 / MAX_NORMALIZATION or kernel_sums_to_zero:
            if nan_interpolate:
                raise ValueError("Setting nan_treatment='interpolate' requires the kernel to be normalized, but the input kernel has a sum close to zero. For a zero-sum kernel and data with NaNs, set nan_treatment='fill'.")
            else:
                raise ValueError(f"The kernel can't be normalized, because its sum is close to zero. The sum of the given kernel is < {1.0 / MAX_NORMALIZATION}")
    if preserve_nan or nan_treatment == 'fill':
        initially_nan = np.isnan(array_internal)
        if nan_treatment == 'fill':
            array_internal[initially_nan] = fill_value
    result = np.zeros(array_internal.shape, dtype=float, order='C')
    embed_result_within_padded_region = True
    array_to_convolve = array_internal
    if boundary in ('fill', 'extend', 'wrap'):
        embed_result_within_padded_region = False
        if boundary == 'fill':
            array_to_convolve = np.full(array_shape + 2 * pad_width, fill_value=fill_value, dtype=float, order='C')
            if array_internal.ndim == 1:
                array_to_convolve[pad_width[0]:array_shape[0] + pad_width[0]] = array_internal
            elif array_internal.ndim == 2:
                array_to_convolve[pad_width[0]:array_shape[0] + pad_width[0], pad_width[1]:array_shape[1] + pad_width[1]] = array_internal
            else:
                array_to_convolve[pad_width[0]:array_shape[0] + pad_width[0], pad_width[1]:array_shape[1] + pad_width[1], pad_width[2]:array_shape[2] + pad_width[2]] = array_internal
        else:
            np_pad_mode_dict = {'fill': 'constant', 'extend': 'edge', 'wrap': 'wrap'}
            np_pad_mode = np_pad_mode_dict[boundary]
            pad_width = kernel_shape // 2
            if array_internal.ndim == 1:
                np_pad_width = (pad_width[0],)
            elif array_internal.ndim == 2:
                np_pad_width = ((pad_width[0],), (pad_width[1],))
            else:
                np_pad_width = ((pad_width[0],), (pad_width[1],), (pad_width[2],))
            array_to_convolve = np.pad(array_internal, pad_width=np_pad_width, mode=np_pad_mode)
    _convolveNd_c(result, array_to_convolve, kernel_internal, nan_interpolate, embed_result_within_padded_region, n_threads)
    if normalize_kernel:
        if not nan_interpolate:
            result /= kernel_sum
    elif nan_interpolate:
        result *= kernel_sum
    if nan_interpolate and (not preserve_nan) and np.isnan(result.sum()):
        warnings.warn("nan_treatment='interpolate', however, NaN values detected post convolution. A contiguous region of NaN values, larger than the kernel size, are present in the input array. Increase the kernel size to avoid this.", AstropyUserWarning)
    if preserve_nan:
        result[initially_nan] = np.nan
    array_unit = getattr(passed_array, 'unit', None)
    if array_unit is not None:
        result <<= array_unit
    if isinstance(passed_array, Kernel):
        if isinstance(passed_array, Kernel1D):
            new_result = Kernel1D(array=result)
        elif isinstance(passed_array, Kernel2D):
            new_result = Kernel2D(array=result)
        else:
            raise TypeError('Only 1D and 2D Kernels are supported.')
        new_result._is_bool = False
        new_result._separable = passed_array._separable
        if isinstance(passed_kernel, Kernel):
            new_result._separable = new_result._separable and passed_kernel._separable
        return new_result
    elif array_dtype.kind == 'f':
        try:
            return result.astype(array_dtype, copy=False)
        except TypeError:
            return result.astype(array_dtype)
    else:
        return result

@support_nddata(data='array')
def convolve_fft(array, kernel, boundary='fill', fill_value=0.0, nan_treatment='interpolate', normalize_kernel=True, normalization_zero_tol=1e-08, preserve_nan=False, mask=None, crop=True, return_fft=False, fft_pad=None, psf_pad=None, min_wt=0.0, allow_huge=False, fftn=np.fft.fftn, ifftn=np.fft.ifftn, complex_dtype=complex, dealias=False):
    if False:
        print('Hello World!')
    '\n    Convolve an ndarray with an nd-kernel.  Returns a convolved image with\n    ``shape = array.shape``.  Assumes kernel is centered.\n\n    `convolve_fft` is very similar to `convolve` in that it replaces ``NaN``\n    values in the original image with interpolated values using the kernel as\n    an interpolation function.  However, it also includes many additional\n    options specific to the implementation.\n\n    `convolve_fft` differs from `scipy.signal.fftconvolve` in a few ways:\n\n    * It can treat ``NaN`` values as zeros or interpolate over them.\n    * ``inf`` values are treated as ``NaN``\n    * It optionally pads to the nearest faster sizes to improve FFT speed.\n      These sizes are optimized for the numpy and scipy implementations, and\n      ``fftconvolve`` uses them by default as well; when using other external\n      functions (see below), results may vary.\n    * Its only valid ``mode`` is \'same\' (i.e., the same shape array is returned)\n    * It lets you use your own fft, e.g.,\n      `pyFFTW <https://pypi.org/project/pyFFTW/>`_ or\n      `pyFFTW3 <https://pypi.org/project/PyFFTW3/0.2.1/>`_ , which can lead to\n      performance improvements, depending on your system configuration.  pyFFTW3\n      is threaded, and therefore may yield significant performance benefits on\n      multi-core machines at the cost of greater memory requirements.  Specify\n      the ``fftn`` and ``ifftn`` keywords to override the default, which is\n      `numpy.fft.fftn` and `numpy.fft.ifftn`.  The `scipy.fft` functions also\n      offer somewhat better performance and a multi-threaded option.\n\n    Parameters\n    ----------\n    array : `numpy.ndarray`\n        Array to be convolved with ``kernel``.  It can be of any\n        dimensionality, though only 1, 2, and 3d arrays have been tested.\n    kernel : `numpy.ndarray` or `astropy.convolution.Kernel`\n        The convolution kernel. The number of dimensions should match those\n        for the array.  The dimensions *do not* have to be odd in all directions,\n        unlike in the non-fft `convolve` function.  The kernel will be\n        normalized if ``normalize_kernel`` is set.  It is assumed to be centered\n        (i.e., shifts may result if your kernel is asymmetric)\n    boundary : {\'fill\', \'wrap\'}, optional\n        A flag indicating how to handle boundaries:\n\n            * \'fill\': set values outside the array boundary to fill_value\n              (default)\n            * \'wrap\': periodic boundary\n\n        The `None` and \'extend\' parameters are not supported for FFT-based\n        convolution.\n    fill_value : float, optional\n        The value to use outside the array when using boundary=\'fill\'.\n    nan_treatment : {\'interpolate\', \'fill\'}, optional\n        The method used to handle NaNs in the input ``array``:\n            * ``\'interpolate\'``: ``NaN`` values are replaced with\n              interpolated values using the kernel as an interpolation\n              function. Note that if the kernel has a sum equal to\n              zero, NaN interpolation is not possible and will raise an\n              exception.\n            * ``\'fill\'``: ``NaN`` values are replaced by ``fill_value``\n              prior to convolution.\n    normalize_kernel : callable or boolean, optional\n        If specified, this is the function to divide kernel by to normalize it.\n        e.g., ``normalize_kernel=np.sum`` means that kernel will be modified to be:\n        ``kernel = kernel / np.sum(kernel)``.  If True, defaults to\n        ``normalize_kernel = np.sum``.\n    normalization_zero_tol : float, optional\n        The absolute tolerance on whether the kernel is different than zero.\n        If the kernel sums to zero to within this precision, it cannot be\n        normalized. Default is "1e-8".\n    preserve_nan : bool, optional\n        After performing convolution, should pixels that were originally NaN\n        again become NaN?\n    mask : None or ndarray, optional\n        A "mask" array.  Shape must match ``array``, and anything that is masked\n        (i.e., not 0/`False`) will be set to NaN for the convolution.  If\n        `None`, no masking will be performed unless ``array`` is a masked array.\n        If ``mask`` is not `None` *and* ``array`` is a masked array, a pixel is\n        masked if it is masked in either ``mask`` *or* ``array.mask``.\n    crop : bool, optional\n        Default on.  Return an image of the size of the larger of the input\n        image and the kernel.\n        If the image and kernel are asymmetric in opposite directions, will\n        return the largest image in both directions.\n        For example, if an input image has shape [100,3] but a kernel with shape\n        [6,6] is used, the output will be [100,6].\n    return_fft : bool, optional\n        Return the ``fft(image)*fft(kernel)`` instead of the convolution (which is\n        ``ifft(fft(image)*fft(kernel))``).  Useful for making PSDs.\n    fft_pad : bool, optional\n        Default on.  Zero-pad image to the nearest size supporting more efficient\n        execution of the FFT, generally values factorizable into the first 3-5\n        prime numbers.  With ``boundary=\'wrap\'``, this will be disabled.\n    psf_pad : bool, optional\n        Zero-pad image to be at least the sum of the image sizes to avoid\n        edge-wrapping when smoothing.  This is enabled by default with\n        ``boundary=\'fill\'``, but it can be overridden with a boolean option.\n        ``boundary=\'wrap\'`` and ``psf_pad=True`` are not compatible.\n    min_wt : float, optional\n        If ignoring ``NaN`` / zeros, force all grid points with a weight less than\n        this value to ``NaN`` (the weight of a grid point with *no* ignored\n        neighbors is 1.0).\n        If ``min_wt`` is zero, then all zero-weight points will be set to zero\n        instead of ``NaN`` (which they would be otherwise, because 1/0 = nan).\n        See the examples below.\n    allow_huge : bool, optional\n        Allow huge arrays in the FFT?  If False, will raise an exception if the\n        array or kernel size is >1 GB.\n    fftn : callable, optional\n        The fft function.  Can be overridden to use your own ffts,\n        e.g. an fftw3 wrapper or scipy\'s fftn, ``fft=scipy.fftpack.fftn``.\n    ifftn : callable, optional\n        The inverse fft function. Can be overridden the same way ``fttn``.\n    complex_dtype : complex type, optional\n        Which complex dtype to use.  `numpy` has a range of options, from 64 to\n        256.\n    dealias: bool, optional\n        Default off. Zero-pad image to enable explicit dealiasing\n        of convolution. With ``boundary=\'wrap\'``, this will be disabled.\n        Note that for an input of nd dimensions this will increase\n        the size of the temporary arrays by at least ``1.5**nd``.\n        This may result in significantly more memory usage.\n\n    Returns\n    -------\n    default : ndarray\n        ``array`` convolved with ``kernel``.  If ``return_fft`` is set, returns\n        ``fft(array) * fft(kernel)``.  If crop is not set, returns the\n        image, but with the fft-padded size instead of the input size.\n\n    Raises\n    ------\n    `ValueError`\n        If the array is bigger than 1 GB after padding, will raise this\n        exception unless ``allow_huge`` is True.\n\n    See Also\n    --------\n    convolve:\n        Convolve is a non-fft version of this code.  It is more memory\n        efficient and for small kernels can be faster.\n\n    Notes\n    -----\n    With ``psf_pad=True`` and a large PSF, the resulting data\n    can become large and consume a lot of memory. See Issue\n    https://github.com/astropy/astropy/pull/4366 and the update in\n    https://github.com/astropy/astropy/pull/11533 for further details.\n\n    Dealiasing of pseudospectral convolutions is necessary for\n    numerical stability of the underlying algorithms. A common\n    method for handling this is to zero pad the image by at least\n    1/2 to eliminate the wavenumbers which have been aliased\n    by convolution. This is so that the aliased 1/3 of the\n    results of the convolution computation can be thrown out. See\n    https://doi.org/10.1175/1520-0469(1971)028%3C1074:OTEOAI%3E2.0.CO;2\n    https://iopscience.iop.org/article/10.1088/1742-6596/318/7/072037\n\n    Note that if dealiasing is necessary to your application, but your\n    process is memory constrained, you may want to consider using\n    FFTW++: https://github.com/dealias/fftwpp. It includes python\n    wrappers for a pseudospectral convolution which will implicitly\n    dealias your convolution without the need for additional padding.\n    Note that one cannot use FFTW++\'s convlution directly in this\n    method as in handles the entire convolution process internally.\n    Additionally, FFTW++ includes other useful pseudospectral methods to\n    consider.\n\n    Examples\n    --------\n    >>> convolve_fft([1, 0, 3], [1, 1, 1])\n    array([0.33333333, 1.33333333, 1.        ])\n\n    >>> convolve_fft([1, np.nan, 3], [1, 1, 1])\n    array([0.5, 2. , 1.5])\n\n    >>> convolve_fft([1, 0, 3], [0, 1, 0])  # doctest: +FLOAT_CMP\n    array([ 1.00000000e+00, -3.70074342e-17,  3.00000000e+00])\n\n    >>> convolve_fft([1, 2, 3], [1])\n    array([1., 2., 3.])\n\n    >>> convolve_fft([1, np.nan, 3], [0, 1, 0], nan_treatment=\'interpolate\')\n    array([1., 0., 3.])\n\n    >>> convolve_fft([1, np.nan, 3], [0, 1, 0], nan_treatment=\'interpolate\',\n    ...              min_wt=1e-8)\n    array([ 1., nan,  3.])\n\n    >>> convolve_fft([1, np.nan, 3], [1, 1, 1], nan_treatment=\'interpolate\')\n    array([0.5, 2. , 1.5])\n\n    >>> convolve_fft([1, np.nan, 3], [1, 1, 1], nan_treatment=\'interpolate\',\n    ...               normalize_kernel=True)\n    array([0.5, 2. , 1.5])\n\n    >>> import scipy.fft  # optional - requires scipy\n    >>> convolve_fft([1, np.nan, 3], [1, 1, 1], nan_treatment=\'interpolate\',\n    ...               normalize_kernel=True,\n    ...               fftn=scipy.fft.fftn, ifftn=scipy.fft.ifftn)\n    array([0.5, 2. , 1.5])\n\n    >>> fft_mp = lambda a: scipy.fft.fftn(a, workers=-1)  # use all available cores\n    >>> ifft_mp = lambda a: scipy.fft.ifftn(a, workers=-1)\n    >>> convolve_fft([1, np.nan, 3], [1, 1, 1], nan_treatment=\'interpolate\',\n    ...               normalize_kernel=True, fftn=fft_mp, ifftn=ifft_mp)\n    array([0.5, 2. , 1.5])\n    '
    if isinstance(kernel, Kernel):
        kernel = kernel.array
        if isinstance(array, Kernel):
            raise TypeError("Can't convolve two kernels with convolve_fft.  Use convolve instead.")
    if nan_treatment not in ('interpolate', 'fill'):
        raise ValueError("nan_treatment must be one of 'interpolate','fill'")
    array_unit = getattr(array, 'unit', None)
    array = _copy_input_if_needed(array, dtype=complex, order='C', nan_treatment=nan_treatment, mask=mask, fill_value=np.nan)
    kernel = _copy_input_if_needed(kernel, dtype=complex, order='C', nan_treatment=None, mask=None, fill_value=0)
    if array.ndim != kernel.ndim:
        raise ValueError('Image and kernel must have same number of dimensions')
    arrayshape = array.shape
    kernshape = kernel.shape
    array_size_B = np.prod(arrayshape, dtype=np.int64) * np.dtype(complex_dtype).itemsize * u.byte
    if array_size_B > 1 * u.GB and (not allow_huge):
        raise ValueError(f'Size Error: Arrays will be {human_file_size(array_size_B)}.  Use allow_huge=True to override this exception.')
    nanmaskarray = np.isnan(array) | np.isinf(array)
    if nan_treatment == 'fill':
        array[nanmaskarray] = fill_value
    else:
        array[nanmaskarray] = 0
    nanmaskkernel = np.isnan(kernel) | np.isinf(kernel)
    kernel[nanmaskkernel] = 0
    if normalize_kernel is True:
        if kernel.sum() < 1.0 / MAX_NORMALIZATION:
            raise Exception(f"The kernel can't be normalized, because its sum is close to zero. The sum of the given kernel is < {1.0 / MAX_NORMALIZATION}")
        kernel_scale = kernel.sum()
        normalized_kernel = kernel / kernel_scale
        kernel_scale = 1
    elif normalize_kernel:
        kernel_scale = normalize_kernel(kernel)
        normalized_kernel = kernel / kernel_scale
    else:
        kernel_scale = kernel.sum()
        if np.abs(kernel_scale) < normalization_zero_tol:
            if nan_treatment == 'interpolate':
                raise ValueError('Cannot interpolate NaNs with an unnormalizable kernel')
            else:
                kernel_scale = 1
                normalized_kernel = kernel
        else:
            normalized_kernel = kernel / kernel_scale
    if boundary is None:
        warnings.warn("The convolve_fft version of boundary=None is equivalent to the convolve boundary='fill'.  There is no FFT equivalent to convolve's zero-if-kernel-leaves-boundary", AstropyUserWarning)
        if psf_pad is None:
            psf_pad = True
        if fft_pad is None:
            fft_pad = True
    elif boundary == 'fill':
        if psf_pad is False:
            warnings.warn(f"psf_pad was set to {psf_pad}, which overrides the boundary='fill' setting.", AstropyUserWarning)
        else:
            psf_pad = True
        if fft_pad is None:
            fft_pad = True
    elif boundary == 'wrap':
        if psf_pad:
            raise ValueError("With boundary='wrap', psf_pad cannot be enabled.")
        psf_pad = False
        if fft_pad:
            raise ValueError("With boundary='wrap', fft_pad cannot be enabled.")
        fft_pad = False
        if dealias:
            raise ValueError("With boundary='wrap', dealias cannot be enabled.")
        fill_value = 0
    elif boundary == 'extend':
        raise NotImplementedError("The 'extend' option is not implemented for fft-based convolution")
    if psf_pad:
        newshape = np.array(arrayshape) + np.array(kernshape)
    else:
        newshape = np.maximum(arrayshape, kernshape)
    if dealias:
        newshape += np.ceil(newshape / 2).astype(int)
    if fft_pad:
        newshape = _next_fast_lengths(newshape)
    array_size_C = np.prod(newshape, dtype=np.int64) * np.dtype(complex_dtype).itemsize * u.byte
    if array_size_C > 1 * u.GB and (not allow_huge):
        raise ValueError(f'Size Error: Arrays will be {human_file_size(array_size_C)}.  Use allow_huge=True to override this exception.')
    arrayslices = []
    kernslices = []
    for (newdimsize, arraydimsize, kerndimsize) in zip(newshape, arrayshape, kernshape):
        center = newdimsize - (newdimsize + 1) // 2
        arrayslices += [slice(center - arraydimsize // 2, center + (arraydimsize + 1) // 2)]
        kernslices += [slice(center - kerndimsize // 2, center + (kerndimsize + 1) // 2)]
    arrayslices = tuple(arrayslices)
    kernslices = tuple(kernslices)
    if not np.all(newshape == arrayshape):
        if np.isfinite(fill_value):
            bigarray = np.ones(newshape, dtype=complex_dtype) * fill_value
        else:
            bigarray = np.zeros(newshape, dtype=complex_dtype)
        bigarray[arrayslices] = array
    else:
        bigarray = array
    if not np.all(newshape == kernshape):
        bigkernel = np.zeros(newshape, dtype=complex_dtype)
        bigkernel[kernslices] = normalized_kernel
    else:
        bigkernel = normalized_kernel
    arrayfft = fftn(bigarray)
    kernfft = fftn(np.fft.ifftshift(bigkernel))
    fftmult = arrayfft * kernfft
    interpolate_nan = nan_treatment == 'interpolate'
    if interpolate_nan:
        if not np.isfinite(fill_value):
            bigimwt = np.zeros(newshape, dtype=complex_dtype)
        else:
            bigimwt = np.ones(newshape, dtype=complex_dtype)
        bigimwt[arrayslices] = 1.0 - nanmaskarray * interpolate_nan
        wtfft = fftn(bigimwt)
        wtfftmult = wtfft * kernfft
        wtsm = ifftn(wtfftmult)
        bigimwt[arrayslices] = wtsm.real[arrayslices]
    else:
        bigimwt = 1
    if np.isnan(fftmult).any():
        raise ValueError('Encountered NaNs in convolve.  This is disallowed.')
    fftmult *= kernel_scale
    if array_unit is not None:
        fftmult <<= array_unit
    if return_fft:
        return fftmult
    if interpolate_nan:
        with np.errstate(divide='ignore', invalid='ignore'):
            rifft = ifftn(fftmult) / bigimwt
        if not np.isscalar(bigimwt):
            if min_wt > 0.0:
                rifft[bigimwt < min_wt] = np.nan
            else:
                rifft[bigimwt < 10 * np.finfo(bigimwt.dtype).eps] = 0.0
    else:
        rifft = ifftn(fftmult)
    if preserve_nan:
        rifft[arrayslices][nanmaskarray] = np.nan
    if crop:
        result = rifft[arrayslices].real
        return result
    else:
        return rifft.real

def interpolate_replace_nans(array, kernel, convolve=convolve, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Given a data set containing NaNs, replace the NaNs by interpolating from\n    neighboring data points with a given kernel.\n\n    Parameters\n    ----------\n    array : `numpy.ndarray`\n        Array to be convolved with ``kernel``.  It can be of any\n        dimensionality, though only 1, 2, and 3d arrays have been tested.\n    kernel : `numpy.ndarray` or `astropy.convolution.Kernel`\n        The convolution kernel. The number of dimensions should match those\n        for the array.  The dimensions *do not* have to be odd in all directions,\n        unlike in the non-fft `convolve` function.  The kernel will be\n        normalized if ``normalize_kernel`` is set.  It is assumed to be centered\n        (i.e., shifts may result if your kernel is asymmetric).  The kernel\n        *must be normalizable* (i.e., its sum cannot be zero).\n    convolve : `convolve` or `convolve_fft`\n        One of the two convolution functions defined in this package.\n\n    Returns\n    -------\n    newarray : `numpy.ndarray`\n        A copy of the original array with NaN pixels replaced with their\n        interpolated counterparts\n    '
    if not np.any(np.isnan(array)):
        return array.copy()
    newarray = array.copy()
    convolved = convolve(array, kernel, nan_treatment='interpolate', normalize_kernel=True, preserve_nan=False, **kwargs)
    isnan = np.isnan(array)
    newarray[isnan] = convolved[isnan]
    return newarray

def convolve_models(model, kernel, mode='convolve_fft', **kwargs):
    if False:
        print('Hello World!')
    "\n    Convolve two models using `~astropy.convolution.convolve_fft`.\n\n    Parameters\n    ----------\n    model : `~astropy.modeling.core.Model`\n        Functional model\n    kernel : `~astropy.modeling.core.Model`\n        Convolution kernel\n    mode : str\n        Keyword representing which function to use for convolution.\n            * 'convolve_fft' : use `~astropy.convolution.convolve_fft` function.\n            * 'convolve' : use `~astropy.convolution.convolve`.\n    **kwargs : dict\n        Keyword arguments to me passed either to `~astropy.convolution.convolve`\n        or `~astropy.convolution.convolve_fft` depending on ``mode``.\n\n    Returns\n    -------\n    default : `~astropy.modeling.core.CompoundModel`\n        Convolved model\n    "
    if mode == 'convolve_fft':
        operator = SPECIAL_OPERATORS.add('convolve_fft', partial(convolve_fft, **kwargs))
    elif mode == 'convolve':
        operator = SPECIAL_OPERATORS.add('convolve', partial(convolve, **kwargs))
    else:
        raise ValueError(f'Mode {mode} is not supported.')
    return CompoundModel(operator, model, kernel)

def convolve_models_fft(model, kernel, bounding_box, resolution, cache=True, **kwargs):
    if False:
        return 10
    '\n    Convolve two models using `~astropy.convolution.convolve_fft`.\n\n    Parameters\n    ----------\n    model : `~astropy.modeling.core.Model`\n        Functional model\n    kernel : `~astropy.modeling.core.Model`\n        Convolution kernel\n    bounding_box : tuple\n        The bounding box which encompasses enough of the support of both\n        the ``model`` and ``kernel`` so that an accurate convolution can be\n        computed.\n    resolution : float\n        The resolution that one wishes to approximate the convolution\n        integral at.\n    cache : optional, bool\n        Default value True. Allow for the storage of the convolution\n        computation for later reuse.\n    **kwargs : dict\n        Keyword arguments to be passed either to `~astropy.convolution.convolve`\n        or `~astropy.convolution.convolve_fft` depending on ``mode``.\n\n    Returns\n    -------\n    default : `~astropy.modeling.core.CompoundModel`\n        Convolved model\n    '
    operator = SPECIAL_OPERATORS.add('convolve_fft', partial(convolve_fft, **kwargs))
    return Convolution(operator, model, kernel, bounding_box, resolution, cache)