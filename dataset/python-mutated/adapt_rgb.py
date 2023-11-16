import functools
import numpy as np
from .. import color
from ..util.dtype import _convert
__all__ = ['adapt_rgb', 'hsv_value', 'each_channel']

def is_rgb_like(image, channel_axis=-1):
    if False:
        while True:
            i = 10
    "Return True if the image *looks* like it's RGB.\n\n    This function should not be public because it is only intended to be used\n    for functions that don't accept volumes as input, since checking an image's\n    shape is fragile.\n    "
    return image.ndim == 3 and image.shape[channel_axis] in (3, 4)

def adapt_rgb(apply_to_rgb):
    if False:
        while True:
            i = 10
    "Return decorator that adapts to RGB images to a gray-scale filter.\n\n    This function is only intended to be used for functions that don't accept\n    volumes as input, since checking an image's shape is fragile.\n\n    Parameters\n    ----------\n    apply_to_rgb : function\n        Function that returns a filtered image from an image-filter and RGB\n        image. This will only be called if the image is RGB-like.\n    "

    def decorator(image_filter):
        if False:
            i = 10
            return i + 15

        @functools.wraps(image_filter)
        def image_filter_adapted(image, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            if is_rgb_like(image):
                return apply_to_rgb(image_filter, image, *args, **kwargs)
            else:
                return image_filter(image, *args, **kwargs)
        return image_filter_adapted
    return decorator

def hsv_value(image_filter, image, *args, **kwargs):
    if False:
        return 10
    'Return color image by applying `image_filter` on HSV-value of `image`.\n\n    Note that this function is intended for use with `adapt_rgb`.\n\n    Parameters\n    ----------\n    image_filter : function\n        Function that filters a gray-scale image.\n    image : array\n        Input image. Note that RGBA images are treated as RGB.\n    '
    hsv = color.rgb2hsv(image[:, :, :3])
    value = hsv[:, :, 2].copy()
    value = image_filter(value, *args, **kwargs)
    hsv[:, :, 2] = _convert(value, hsv.dtype)
    return color.hsv2rgb(hsv)

def each_channel(image_filter, image, *args, **kwargs):
    if False:
        while True:
            i = 10
    'Return color image by applying `image_filter` on channels of `image`.\n\n    Note that this function is intended for use with `adapt_rgb`.\n\n    Parameters\n    ----------\n    image_filter : function\n        Function that filters a gray-scale image.\n    image : array\n        Input image.\n    '
    c_new = [image_filter(c, *args, **kwargs) for c in np.moveaxis(image, -1, 0)]
    return np.stack(c_new, axis=-1)