import numpy as np
from .._shared import utils

def _match_cumulative_cdf(source, template):
    if False:
        i = 10
        return i + 15
    '\n    Return modified source array so that the cumulative density function of\n    its values matches the cumulative density function of the template.\n    '
    if source.dtype.kind == 'u':
        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template.reshape(-1))
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        (src_values, src_lookup, src_counts) = np.unique(source.reshape(-1), return_inverse=True, return_counts=True)
        (tmpl_values, tmpl_counts) = np.unique(template.reshape(-1), return_counts=True)
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(source.shape)

@utils.channel_as_last_axis(channel_arg_positions=(0, 1))
def match_histograms(image, reference, *, channel_axis=None):
    if False:
        i = 10
        return i + 15
    'Adjust an image so that its cumulative histogram matches that of another.\n\n    The adjustment is applied separately for each channel.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image. Can be gray-scale or in color.\n    reference : ndarray\n        Image to match histogram of. Must have the same number of channels as\n        image.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n    Returns\n    -------\n    matched : ndarray\n        Transformed input image.\n\n    Raises\n    ------\n    ValueError\n        Thrown when the number of channels in the input image and the reference\n        differ.\n\n    References\n    ----------\n    .. [1] http://paulbourke.net/miscellaneous/equalisation/\n\n    '
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')
    if channel_axis is not None:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference image must match!')
        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)
    if matched.dtype.kind == 'f':
        out_dtype = utils._supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched