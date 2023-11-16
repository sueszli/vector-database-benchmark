from collections import namedtuple
import numpy as np
from ...util import dtype as dtypes
from ...exposure import is_low_contrast
from ..._shared.utils import warn
from math import floor, ceil
_default_colormap = 'gray'
_nonstandard_colormap = 'viridis'
_diverging_colormap = 'RdBu'
ImageProperties = namedtuple('ImageProperties', ['signed', 'out_of_range_float', 'low_data_range', 'unsupported_dtype'])

def _get_image_properties(image):
    if False:
        while True:
            i = 10
    'Determine nonstandard properties of an input image.\n\n    Parameters\n    ----------\n    image : array\n        The input image.\n\n    Returns\n    -------\n    ip : ImageProperties named tuple\n        The properties of the image:\n\n        - signed: whether the image has negative values.\n        - out_of_range_float: if the image has floating point data\n          outside of [-1, 1].\n        - low_data_range: if the image is in the standard image\n          range (e.g. [0, 1] for a floating point image) but its\n          data range would be too small to display with standard\n          image ranges.\n        - unsupported_dtype: if the image data type is not a\n          standard skimage type, e.g. ``numpy.uint64``.\n    '
    (immin, immax) = (np.min(image), np.max(image))
    imtype = image.dtype.type
    try:
        (lo, hi) = dtypes.dtype_range[imtype]
    except KeyError:
        (lo, hi) = (immin, immax)
    signed = immin < 0
    out_of_range_float = np.issubdtype(image.dtype, np.floating) and (immin < lo or immax > hi)
    low_data_range = immin != immax and is_low_contrast(image)
    unsupported_dtype = image.dtype not in dtypes._supported_types
    return ImageProperties(signed, out_of_range_float, low_data_range, unsupported_dtype)

def _raise_warnings(image_properties):
    if False:
        i = 10
        return i + 15
    'Raise the appropriate warning for each nonstandard image type.\n\n    Parameters\n    ----------\n    image_properties : ImageProperties named tuple\n        The properties of the considered image.\n    '
    ip = image_properties
    if ip.unsupported_dtype:
        warn('Non-standard image type; displaying image with stretched contrast.', stacklevel=3)
    if ip.low_data_range:
        warn('Low image data range; displaying image with stretched contrast.', stacklevel=3)
    if ip.out_of_range_float:
        warn('Float image out of standard range; displaying image with stretched contrast.', stacklevel=3)

def _get_display_range(image):
    if False:
        i = 10
        return i + 15
    'Return the display range for a given set of image properties.\n\n    Parameters\n    ----------\n    image : array\n        The input image.\n\n    Returns\n    -------\n    lo, hi : same type as immin, immax\n        The display range to be used for the input image.\n    cmap : string\n        The name of the colormap to use.\n    '
    ip = _get_image_properties(image)
    (immin, immax) = (np.min(image), np.max(image))
    if ip.signed:
        magnitude = max(abs(immin), abs(immax))
        (lo, hi) = (-magnitude, magnitude)
        cmap = _diverging_colormap
    elif any(ip):
        _raise_warnings(ip)
        (lo, hi) = (immin, immax)
        cmap = _nonstandard_colormap
    else:
        lo = 0
        imtype = image.dtype.type
        hi = dtypes.dtype_range[imtype][1]
        cmap = _default_colormap
    return (lo, hi, cmap)

def imshow(image, ax=None, show_cbar=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Show the input image and return the current axes.\n\n    By default, the image is displayed in grayscale, rather than\n    the matplotlib default colormap.\n\n    Images are assumed to have standard range for their type. For\n    example, if a floating point image has values in [0, 0.5], the\n    most intense color will be gray50, not white.\n\n    If the image exceeds the standard range, or if the range is too\n    small to display, we fall back on displaying exactly the range of\n    the input image, along with a colorbar to clearly indicate that\n    this range transformation has occurred.\n\n    For signed images, we use a diverging colormap centered at 0.\n\n    Parameters\n    ----------\n    image : array, shape (M, N[, 3])\n        The image to display.\n    ax : `matplotlib.axes.Axes`, optional\n        The axis to use for the image, defaults to plt.gca().\n    show_cbar : boolean, optional.\n        Whether to show the colorbar (used to override default behavior).\n    **kwargs : Keyword arguments\n        These are passed directly to `matplotlib.pyplot.imshow`.\n\n    Returns\n    -------\n    ax_im : `matplotlib.pyplot.AxesImage`\n        The `AxesImage` object returned by `plt.imshow`.\n    '
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    (lo, hi, cmap) = _get_display_range(image)
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', cmap)
    kwargs.setdefault('vmin', lo)
    kwargs.setdefault('vmax', hi)
    ax = ax or plt.gca()
    ax_im = ax.imshow(image, **kwargs)
    if cmap != _default_colormap and show_cbar is not False or show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(ax_im, cax=cax)
    ax.get_figure().tight_layout()
    return ax_im

def imshow_collection(ic, *args, **kwargs):
    if False:
        while True:
            i = 10
    'Display all images in the collection.\n\n    Returns\n    -------\n    fig : `matplotlib.figure.Figure`\n        The `Figure` object returned by `plt.subplots`.\n    '
    import matplotlib.pyplot as plt
    if len(ic) < 1:
        raise ValueError('Number of images to plot must be greater than 0')
    num_images = len(ic)
    k = (num_images * 12) ** 0.5
    r1 = max(1, floor(k / 4))
    r2 = ceil(k / 4)
    c1 = ceil(num_images / r1)
    c2 = ceil(num_images / r2)
    if abs(r1 / c1 - 0.75) < abs(r2 / c2 - 0.75):
        (nrows, ncols) = (r1, c1)
    else:
        (nrows, ncols) = (r2, c2)
    (fig, axes) = plt.subplots(nrows=nrows, ncols=ncols)
    ax = np.asarray(axes).ravel()
    for (n, image) in enumerate(ic):
        ax[n].imshow(image, *args, **kwargs)
    kwargs['ax'] = axes
    return fig

def imread(*args, **kwargs):
    if False:
        return 10
    import matplotlib.image
    return matplotlib.image.imread(*args, **kwargs)

def _app_show():
    if False:
        i = 10
        return i + 15
    from matplotlib.pyplot import show
    show()