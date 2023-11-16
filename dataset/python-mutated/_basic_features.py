from itertools import combinations_with_replacement
import itertools
import numpy as np
from skimage import filters, feature
from skimage.util.dtype import img_as_float32
from concurrent.futures import ThreadPoolExecutor

def _texture_filter(gaussian_filtered):
    if False:
        for i in range(10):
            print('nop')
    H_elems = [np.gradient(np.gradient(gaussian_filtered)[ax0], axis=ax1) for (ax0, ax1) in combinations_with_replacement(range(gaussian_filtered.ndim), 2)]
    eigvals = feature.hessian_matrix_eigvals(H_elems)
    return eigvals

def _singlescale_basic_features_singlechannel(img, sigma, intensity=True, edges=True, texture=True):
    if False:
        return 10
    results = ()
    gaussian_filtered = filters.gaussian(img, sigma, preserve_range=False)
    if intensity:
        results += (gaussian_filtered,)
    if edges:
        results += (filters.sobel(gaussian_filtered),)
    if texture:
        results += (*_texture_filter(gaussian_filtered),)
    return results

def _mutiscale_basic_features_singlechannel(img, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16, num_sigma=None, num_workers=None):
    if False:
        i = 10
        return i + 15
    'Features for a single channel nd image.\n\n    Parameters\n    ----------\n    img : ndarray\n        Input image, which can be grayscale or multichannel.\n    intensity : bool, default True\n        If True, pixel intensities averaged over the different scales\n        are added to the feature set.\n    edges : bool, default True\n        If True, intensities of local gradients averaged over the different\n        scales are added to the feature set.\n    texture : bool, default True\n        If True, eigenvalues of the Hessian matrix after Gaussian blurring\n        at different scales are added to the feature set.\n    sigma_min : float, optional\n        Smallest value of the Gaussian kernel used to average local\n        neighborhoods before extracting features.\n    sigma_max : float, optional\n        Largest value of the Gaussian kernel used to average local\n        neighborhoods before extracting features.\n    num_sigma : int, optional\n        Number of values of the Gaussian kernel between sigma_min and sigma_max.\n        If None, sigma_min multiplied by powers of 2 are used.\n    num_workers : int or None, optional\n        The number of parallel threads to use. If set to ``None``, the full\n        set of available cores are used.\n\n    Returns\n    -------\n    features : list\n        List of features, each element of the list is an array of shape as img.\n    '
    img = np.ascontiguousarray(img_as_float32(img))
    if num_sigma is None:
        num_sigma = int(np.log2(sigma_max) - np.log2(sigma_min) + 1)
    sigmas = np.logspace(np.log2(sigma_min), np.log2(sigma_max), num=num_sigma, base=2, endpoint=True)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        out_sigmas = list(ex.map(lambda s: _singlescale_basic_features_singlechannel(img, s, intensity=intensity, edges=edges, texture=texture), sigmas))
    features = itertools.chain.from_iterable(out_sigmas)
    return features

def multiscale_basic_features(image, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16, num_sigma=None, num_workers=None, *, channel_axis=None):
    if False:
        while True:
            i = 10
    'Local features for a single- or multi-channel nd image.\n\n    Intensity, gradient intensity and local structure are computed at\n    different scales thanks to Gaussian blurring.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image, which can be grayscale or multichannel.\n    intensity : bool, default True\n        If True, pixel intensities averaged over the different scales\n        are added to the feature set.\n    edges : bool, default True\n        If True, intensities of local gradients averaged over the different\n        scales are added to the feature set.\n    texture : bool, default True\n        If True, eigenvalues of the Hessian matrix after Gaussian blurring\n        at different scales are added to the feature set.\n    sigma_min : float, optional\n        Smallest value of the Gaussian kernel used to average local\n        neighborhoods before extracting features.\n    sigma_max : float, optional\n        Largest value of the Gaussian kernel used to average local\n        neighborhoods before extracting features.\n    num_sigma : int, optional\n        Number of values of the Gaussian kernel between sigma_min and sigma_max.\n        If None, sigma_min multiplied by powers of 2 are used.\n    num_workers : int or None, optional\n        The number of parallel threads to use. If set to ``None``, the full\n        set of available cores are used.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    features : np.ndarray\n        Array of shape ``image.shape + (n_features,)``. When `channel_axis` is\n        not None, all channels are concatenated along the features dimension.\n        (i.e. ``n_features == n_features_singlechannel * n_channels``)\n    '
    if not any([intensity, edges, texture]):
        raise ValueError('At least one of `intensity`, `edges` or `textures`must be True for features to be computed.')
    if channel_axis is None:
        image = image[..., np.newaxis]
        channel_axis = -1
    elif channel_axis != -1:
        image = np.moveaxis(image, channel_axis, -1)
    all_results = (_mutiscale_basic_features_singlechannel(image[..., dim], intensity=intensity, edges=edges, texture=texture, sigma_min=sigma_min, sigma_max=sigma_max, num_sigma=num_sigma, num_workers=num_workers) for dim in range(image.shape[-1]))
    features = list(itertools.chain.from_iterable(all_results))
    out = np.stack(features, axis=-1)
    return out