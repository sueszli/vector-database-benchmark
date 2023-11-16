import math
import numpy as np
from .._shared.filters import gaussian
from .._shared.utils import convert_to_float
from ._warps import resize

def _smooth(image, sigma, mode, cval, channel_axis):
    if False:
        i = 10
        return i + 15
    'Return image with each channel smoothed by the Gaussian filter.'
    smoothed = np.empty_like(image)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        sigma = (sigma,) * (image.ndim - 1)
    else:
        channel_axis = None
    gaussian(image, sigma, output=smoothed, mode=mode, cval=cval, channel_axis=channel_axis)
    return smoothed

def _check_factor(factor):
    if False:
        i = 10
        return i + 15
    if factor <= 1:
        raise ValueError('scale factor must be greater than 1')

def pyramid_reduce(image, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None):
    if False:
        i = 10
        return i + 15
    "Smooth and then downsample image.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    downscale : float, optional\n        Downscale factor.\n    sigma : float, optional\n        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which\n        corresponds to a filter mask twice the size of the scale factor that\n        covers more than 99% of the Gaussian distribution.\n    order : int, optional\n        Order of splines used in interpolation of downsampling. See\n        `skimage.transform.warp` for detail.\n    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional\n        The mode parameter determines how the array borders are handled, where\n        cval is the value when mode is equal to 'constant'.\n    cval : float, optional\n        Value to fill past edges of input if mode is 'constant'.\n    preserve_range : bool, optional\n        Whether to keep the original range of values. Otherwise, the input\n        image is converted according to the conventions of `img_as_float`.\n        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : array\n        Smoothed and downsampled float image.\n\n    References\n    ----------\n    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf\n\n    "
    _check_factor(downscale)
    image = convert_to_float(image, preserve_range)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        out_shape = tuple((math.ceil(d / float(downscale)) if ax != channel_axis else d for (ax, d) in enumerate(image.shape)))
    else:
        out_shape = tuple((math.ceil(d / float(downscale)) for d in image.shape))
    if sigma is None:
        sigma = 2 * downscale / 6.0
    smoothed = _smooth(image, sigma, mode, cval, channel_axis)
    out = resize(smoothed, out_shape, order=order, mode=mode, cval=cval, anti_aliasing=False)
    return out

def pyramid_expand(image, upscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None):
    if False:
        return 10
    "Upsample and then smooth image.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    upscale : float, optional\n        Upscale factor.\n    sigma : float, optional\n        Sigma for Gaussian filter. Default is `2 * upscale / 6.0` which\n        corresponds to a filter mask twice the size of the scale factor that\n        covers more than 99% of the Gaussian distribution.\n    order : int, optional\n        Order of splines used in interpolation of upsampling. See\n        `skimage.transform.warp` for detail.\n    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional\n        The mode parameter determines how the array borders are handled, where\n        cval is the value when mode is equal to 'constant'.\n    cval : float, optional\n        Value to fill past edges of input if mode is 'constant'.\n    preserve_range : bool, optional\n        Whether to keep the original range of values. Otherwise, the input\n        image is converted according to the conventions of `img_as_float`.\n        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : array\n        Upsampled and smoothed float image.\n\n    References\n    ----------\n    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf\n\n    "
    _check_factor(upscale)
    image = convert_to_float(image, preserve_range)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        out_shape = tuple((math.ceil(upscale * d) if ax != channel_axis else d for (ax, d) in enumerate(image.shape)))
    else:
        out_shape = tuple((math.ceil(upscale * d) for d in image.shape))
    if sigma is None:
        sigma = 2 * upscale / 6.0
    resized = resize(image, out_shape, order=order, mode=mode, cval=cval, anti_aliasing=False)
    out = _smooth(resized, sigma, mode, cval, channel_axis)
    return out

def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None):
    if False:
        return 10
    "Yield images of the Gaussian pyramid formed by the input image.\n\n    Recursively applies the `pyramid_reduce` function to the image, and yields\n    the downscaled images.\n\n    Note that the first image of the pyramid will be the original, unscaled\n    image. The total number of images is `max_layer + 1`. In case all layers\n    are computed, the last image is either a one-pixel image or the image where\n    the reduction does not change its shape.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    max_layer : int, optional\n        Number of layers for the pyramid. 0th layer is the original image.\n        Default is -1 which builds all possible layers.\n    downscale : float, optional\n        Downscale factor.\n    sigma : float, optional\n        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which\n        corresponds to a filter mask twice the size of the scale factor that\n        covers more than 99% of the Gaussian distribution.\n    order : int, optional\n        Order of splines used in interpolation of downsampling. See\n        `skimage.transform.warp` for detail.\n    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional\n        The mode parameter determines how the array borders are handled, where\n        cval is the value when mode is equal to 'constant'.\n    cval : float, optional\n        Value to fill past edges of input if mode is 'constant'.\n    preserve_range : bool, optional\n        Whether to keep the original range of values. Otherwise, the input\n        image is converted according to the conventions of `img_as_float`.\n        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    pyramid : generator\n        Generator yielding pyramid layers as float images.\n\n    References\n    ----------\n    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf\n\n    "
    _check_factor(downscale)
    image = convert_to_float(image, preserve_range)
    layer = 0
    current_shape = image.shape
    prev_layer_image = image
    yield image
    while layer != max_layer:
        layer += 1
        layer_image = pyramid_reduce(prev_layer_image, downscale, sigma, order, mode, cval, channel_axis=channel_axis)
        prev_shape = current_shape
        prev_layer_image = layer_image
        current_shape = layer_image.shape
        if current_shape == prev_shape:
            break
        yield layer_image

def pyramid_laplacian(image, max_layer=-1, downscale=2, sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, channel_axis=None):
    if False:
        while True:
            i = 10
    "Yield images of the laplacian pyramid formed by the input image.\n\n    Each layer contains the difference between the downsampled and the\n    downsampled, smoothed image::\n\n        layer = resize(prev_layer) - smooth(resize(prev_layer))\n\n    Note that the first image of the pyramid will be the difference between the\n    original, unscaled image and its smoothed version. The total number of\n    images is `max_layer + 1`. In case all layers are computed, the last image\n    is either a one-pixel image or the image where the reduction does not\n    change its shape.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    max_layer : int, optional\n        Number of layers for the pyramid. 0th layer is the original image.\n        Default is -1 which builds all possible layers.\n    downscale : float, optional\n        Downscale factor.\n    sigma : float, optional\n        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which\n        corresponds to a filter mask twice the size of the scale factor that\n        covers more than 99% of the Gaussian distribution.\n    order : int, optional\n        Order of splines used in interpolation of downsampling. See\n        `skimage.transform.warp` for detail.\n    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional\n        The mode parameter determines how the array borders are handled, where\n        cval is the value when mode is equal to 'constant'.\n    cval : float, optional\n        Value to fill past edges of input if mode is 'constant'.\n    preserve_range : bool, optional\n        Whether to keep the original range of values. Otherwise, the input\n        image is converted according to the conventions of `img_as_float`.\n        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    pyramid : generator\n        Generator yielding pyramid layers as float images.\n\n    References\n    ----------\n    .. [1] http://persci.mit.edu/pub_pdfs/pyramid83.pdf\n    .. [2] http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html\n\n    "
    _check_factor(downscale)
    image = convert_to_float(image, preserve_range)
    if sigma is None:
        sigma = 2 * downscale / 6.0
    current_shape = image.shape
    smoothed_image = _smooth(image, sigma, mode, cval, channel_axis)
    yield (image - smoothed_image)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        shape_without_channels = list(current_shape)
        shape_without_channels.pop(channel_axis)
        shape_without_channels = tuple(shape_without_channels)
    else:
        shape_without_channels = current_shape
    if max_layer == -1:
        max_layer = math.ceil(math.log(max(shape_without_channels), downscale))
    for layer in range(max_layer):
        if channel_axis is not None:
            out_shape = tuple((math.ceil(d / float(downscale)) if ax != channel_axis else d for (ax, d) in enumerate(current_shape)))
        else:
            out_shape = tuple((math.ceil(d / float(downscale)) for d in current_shape))
        resized_image = resize(smoothed_image, out_shape, order=order, mode=mode, cval=cval, anti_aliasing=False)
        smoothed_image = _smooth(resized_image, sigma, mode, cval, channel_axis)
        current_shape = resized_image.shape
        yield (resized_image - smoothed_image)