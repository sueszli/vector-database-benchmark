import itertools
import functools
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..metrics import mean_squared_error
from ..util import img_as_float

def _interpolate_image(image, *, multichannel=False):
    if False:
        for i in range(10):
            print('nop')
    'Replacing each pixel in ``image`` with the average of its neighbors.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input data to be interpolated.\n    multichannel : bool, optional\n        Whether the last axis of the image is to be interpreted as multiple\n        channels or another spatial dimension.\n\n    Returns\n    -------\n    interp : ndarray\n        Interpolated version of `image`.\n    '
    spatialdims = image.ndim if not multichannel else image.ndim - 1
    conv_filter = ndi.generate_binary_structure(spatialdims, 1).astype(image.dtype)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()
    if multichannel:
        interp = np.zeros_like(image)
        for i in range(image.shape[-1]):
            interp[..., i] = ndi.convolve(image[..., i], conv_filter, mode='mirror')
    else:
        interp = ndi.convolve(image, conv_filter, mode='mirror')
    return interp

def _generate_grid_slice(shape, *, offset, stride=3):
    if False:
        for i in range(10):
            print('nop')
    'Generate slices of uniformly-spaced points in an array.\n\n    Parameters\n    ----------\n    shape : tuple of int\n        Shape of the mask.\n    offset : int\n        The offset of the grid of ones. Iterating over ``offset`` will cover\n        the entire array. It should be between 0 and ``stride ** ndim``, not\n        inclusive, where ``ndim = len(shape)``.\n    stride : int, optional\n        The spacing between ones, used in each dimension.\n\n    Returns\n    -------\n    mask : ndarray\n        The mask.\n\n    Examples\n    --------\n    >>> shape = (4, 4)\n    >>> array = np.zeros(shape, dtype=int)\n    >>> grid_slice = _generate_grid_slice(shape, offset=0, stride=2)\n    >>> array[grid_slice] = 1\n    >>> print(array)\n    [[1 0 1 0]\n     [0 0 0 0]\n     [1 0 1 0]\n     [0 0 0 0]]\n\n    Changing the offset moves the location of the 1s:\n\n    >>> array = np.zeros(shape, dtype=int)\n    >>> grid_slice = _generate_grid_slice(shape, offset=3, stride=2)\n    >>> array[grid_slice] = 1\n    >>> print(array)\n    [[0 0 0 0]\n     [0 1 0 1]\n     [0 0 0 0]\n     [0 1 0 1]]\n    '
    phases = np.unravel_index(offset, (stride,) * len(shape))
    mask = tuple((slice(p, None, stride) for p in phases))
    return mask

def denoise_invariant(image, denoise_function, *, stride=4, masks=None, denoiser_kwargs=None):
    if False:
        for i in range(10):
            print('nop')
    'Apply a J-invariant version of a denoising function.\n\n    Parameters\n    ----------\n    image : ndarray (M[, N[, ...]][, C]) of ints, uints or floats\n        Input data to be denoised. `image` can be of any numeric type,\n        but it is cast into a ndarray of floats (using `img_as_float`) for the\n        computation of the denoised image.\n    denoise_function : function\n        Original denoising function.\n    stride : int, optional\n        Stride used in masking procedure that converts `denoise_function`\n        to J-invariance.\n    masks : list of ndarray, optional\n        Set of masks to use for computing J-invariant output. If `None`,\n        a full set of masks covering the image will be used.\n    denoiser_kwargs:\n        Keyword arguments passed to `denoise_function`.\n\n    Returns\n    -------\n    output : ndarray\n        Denoised image, of same shape as `image`.\n\n    Notes\n    -----\n    A denoising function is J-invariant if the prediction it makes for each\n    pixel does not depend on the value of that pixel in the original image.\n    The prediction for each pixel may instead use all the relevant information\n    contained in the rest of the image, which is typically quite significant.\n    Any function can be converted into a J-invariant one using a simple masking\n    procedure, as described in [1].\n\n    The pixel-wise error of a J-invariant denoiser is uncorrelated to the noise,\n    so long as the noise in each pixel is independent. Consequently, the average\n    difference between the denoised image and the oisy image, the\n    *self-supervised loss*, is the same as the difference between the denoised\n    image and the original clean image, the *ground-truth loss* (up to a\n    constant).\n\n    This means that the best J-invariant denoiser for a given image can be found\n    using the noisy data alone, by selecting the denoiser minimizing the self-\n    supervised loss.\n\n    References\n    ----------\n    .. [1] J. Batson & L. Royer. Noise2Self: Blind Denoising by Self-Supervision,\n       International Conference on Machine Learning, p. 524-533 (2019).\n\n    Examples\n    --------\n    >>> import skimage\n    >>> from skimage.restoration import denoise_invariant, denoise_tv_chambolle\n    >>> image = skimage.util.img_as_float(skimage.data.chelsea())\n    >>> noisy = skimage.util.random_noise(image, var=0.2 ** 2)\n    >>> denoised = denoise_invariant(noisy, denoise_function=denoise_tv_chambolle)\n    '
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if denoiser_kwargs is None:
        denoiser_kwargs = {}
    multichannel = denoiser_kwargs.get('channel_axis', None) is not None
    interp = _interpolate_image(image, multichannel=multichannel)
    output = np.zeros_like(image)
    if masks is None:
        spatialdims = image.ndim if not multichannel else image.ndim - 1
        n_masks = stride ** spatialdims
        masks = (_generate_grid_slice(image.shape[:spatialdims], offset=idx, stride=stride) for idx in range(n_masks))
    for mask in masks:
        input_image = image.copy()
        input_image[mask] = interp[mask]
        output[mask] = denoise_function(input_image, **denoiser_kwargs)[mask]
    return output

def _product_from_dict(dictionary):
    if False:
        while True:
            i = 10
    'Utility function to convert parameter ranges to parameter combinations.\n\n    Converts a dict of lists into a list of dicts whose values consist of the\n    cartesian product of the values in the original dict.\n\n    Parameters\n    ----------\n    dictionary : dict of lists\n        Dictionary of lists to be multiplied.\n\n    Yields\n    ------\n    selections : dicts of values\n        Dicts containing individual combinations of the values in the input\n        dict.\n    '
    keys = dictionary.keys()
    for element in itertools.product(*dictionary.values()):
        yield dict(zip(keys, element))

def calibrate_denoiser(image, denoise_function, denoise_parameters, *, stride=4, approximate_loss=True, extra_output=False):
    if False:
        print('Hello World!')
    "Calibrate a denoising function and return optimal J-invariant version.\n\n    The returned function is partially evaluated with optimal parameter values\n    set for denoising the input image.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input data to be denoised (converted using `img_as_float`).\n    denoise_function : function\n        Denoising function to be calibrated.\n    denoise_parameters : dict of list\n        Ranges of parameters for `denoise_function` to be calibrated over.\n    stride : int, optional\n        Stride used in masking procedure that converts `denoise_function`\n        to J-invariance.\n    approximate_loss : bool, optional\n        Whether to approximate the self-supervised loss used to evaluate the\n        denoiser by only computing it on one masked version of the image.\n        If False, the runtime will be a factor of `stride**image.ndim` longer.\n    extra_output : bool, optional\n        If True, return parameters and losses in addition to the calibrated\n        denoising function\n\n    Returns\n    -------\n    best_denoise_function : function\n        The optimal J-invariant version of `denoise_function`.\n\n    If `extra_output` is True, the following tuple is also returned:\n\n    (parameters_tested, losses) : tuple (list of dict, list of int)\n        List of parameters tested for `denoise_function`, as a dictionary of\n        kwargs\n        Self-supervised loss for each set of parameters in `parameters_tested`.\n\n\n    Notes\n    -----\n\n    The calibration procedure uses a self-supervised mean-square-error loss\n    to evaluate the performance of J-invariant versions of `denoise_function`.\n    The minimizer of the self-supervised loss is also the minimizer of the\n    ground-truth loss (i.e., the true MSE error) [1]. The returned function\n    can be used on the original noisy image, or other images with similar\n    characteristics.\n\n    Increasing the stride increases the performance of `best_denoise_function`\n     at the expense of increasing its runtime. It has no effect on the runtime\n     of the calibration.\n\n    References\n    ----------\n    .. [1] J. Batson & L. Royer. Noise2Self: Blind Denoising by Self-Supervision,\n           International Conference on Machine Learning, p. 524-533 (2019).\n\n    Examples\n    --------\n    >>> from skimage import color, data\n    >>> from skimage.restoration import denoise_tv_chambolle\n    >>> import numpy as np\n    >>> img = color.rgb2gray(data.astronaut()[:50, :50])\n    >>> rng = np.random.default_rng()\n    >>> noisy = img + 0.5 * img.std() * rng.standard_normal(img.shape)\n    >>> parameters = {'weight': np.arange(0.01, 0.3, 0.02)}\n    >>> denoising_function = calibrate_denoiser(noisy, denoise_tv_chambolle,\n    ...                                         denoise_parameters=parameters)\n    >>> denoised_img = denoising_function(img)\n\n    "
    (parameters_tested, losses) = _calibrate_denoiser_search(image, denoise_function, denoise_parameters=denoise_parameters, stride=stride, approximate_loss=approximate_loss)
    idx = np.argmin(losses)
    best_parameters = parameters_tested[idx]
    best_denoise_function = functools.partial(denoise_invariant, denoise_function=denoise_function, stride=stride, denoiser_kwargs=best_parameters)
    if extra_output:
        return (best_denoise_function, (parameters_tested, losses))
    else:
        return best_denoise_function

def _calibrate_denoiser_search(image, denoise_function, denoise_parameters, *, stride=4, approximate_loss=True):
    if False:
        print('Hello World!')
    'Return a parameter search history with losses for a denoise function.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input data to be denoised (converted using `img_as_float`).\n    denoise_function : function\n        Denoising function to be calibrated.\n    denoise_parameters : dict of list\n        Ranges of parameters for `denoise_function` to be calibrated over.\n    stride : int, optional\n        Stride used in masking procedure that converts `denoise_function`\n        to J-invariance.\n    approximate_loss : bool, optional\n        Whether to approximate the self-supervised loss used to evaluate the\n        denoiser by only computing it on one masked version of the image.\n        If False, the runtime will be a factor of `stride**image.ndim` longer.\n\n    Returns\n    -------\n    parameters_tested : list of dict\n        List of parameters tested for `denoise_function`, as a dictionary of\n        kwargs.\n    losses : list of int\n        Self-supervised loss for each set of parameters in `parameters_tested`.\n    '
    image = img_as_float(image)
    parameters_tested = list(_product_from_dict(denoise_parameters))
    losses = []
    for denoiser_kwargs in parameters_tested:
        multichannel = denoiser_kwargs.get('channel_axis', None) is not None
        if not approximate_loss:
            denoised = denoise_invariant(image, denoise_function, stride=stride, denoiser_kwargs=denoiser_kwargs)
            loss = mean_squared_error(image, denoised)
        else:
            spatialdims = image.ndim if not multichannel else image.ndim - 1
            n_masks = stride ** spatialdims
            mask = _generate_grid_slice(image.shape[:spatialdims], offset=n_masks // 2, stride=stride)
            masked_denoised = denoise_invariant(image, denoise_function, masks=[mask], denoiser_kwargs=denoiser_kwargs)
            loss = mean_squared_error(image[mask], masked_denoised[mask])
        losses.append(loss)
    return (parameters_tested, losses)