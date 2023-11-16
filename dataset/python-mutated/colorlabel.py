import itertools
import numpy as np
from .._shared.utils import _supported_float_type, warn
from ..util import img_as_float
from . import rgb_colors
from .colorconv import gray2rgb, rgb2hsv, hsv2rgb
__all__ = ['color_dict', 'label2rgb', 'DEFAULT_COLORS']
DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')
color_dict = {k: v for (k, v) in rgb_colors.__dict__.items() if isinstance(v, tuple)}

def _rgb_vector(color):
    if False:
        return 10
    'Return RGB color as (1, 3) array.\n\n    This RGB array gets multiplied by masked regions of an RGB image, which are\n    partially flattened by masking (i.e. dimensions 2D + RGB -> 1D + RGB).\n\n    Parameters\n    ----------\n    color : str or array\n        Color name in ``skimage.color.color_dict`` or RGB float values between [0, 1].\n    '
    if isinstance(color, str):
        color = color_dict[color]
    return np.array(color[:3])

def _match_label_with_color(label, colors, bg_label, bg_color):
    if False:
        print('Hello World!')
    'Return `unique_labels` and `color_cycle` for label array and color list.\n\n    Colors are cycled for normal labels, but the background color should only\n    be used for the background.\n    '
    if bg_color is None:
        bg_color = (0, 0, 0)
    bg_color = _rgb_vector(bg_color)
    (unique_labels, mapped_labels) = np.unique(label, return_inverse=True)
    bg_label_rank_list = mapped_labels[label.flat == bg_label]
    if len(bg_label_rank_list) > 0:
        bg_label_rank = bg_label_rank_list[0]
        mapped_labels[mapped_labels < bg_label_rank] += 1
        mapped_labels[label.flat == bg_label] = 0
    else:
        mapped_labels += 1
    color_cycle = itertools.cycle(colors)
    color_cycle = itertools.chain([bg_color], color_cycle)
    return (mapped_labels, color_cycle)

def label2rgb(label, image=None, colors=None, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay', *, saturation=0, channel_axis=-1):
    if False:
        while True:
            i = 10
    "Return an RGB image where color-coded labels are painted over the image.\n\n    Parameters\n    ----------\n    label : ndarray\n        Integer array of labels with the same shape as `image`.\n    image : ndarray, optional\n        Image used as underlay for labels. It should have the same shape as\n        `labels`, optionally with an additional RGB (channels) axis. If `image`\n        is an RGB image, it is converted to grayscale before coloring.\n    colors : list, optional\n        List of colors. If the number of labels exceeds the number of colors,\n        then the colors are cycled.\n    alpha : float [0, 1], optional\n        Opacity of colorized labels. Ignored if image is `None`.\n    bg_label : int, optional\n        Label that's treated as the background. If `bg_label` is specified,\n        `bg_color` is `None`, and `kind` is `overlay`,\n        background is not painted by any colors.\n    bg_color : str or array, optional\n        Background color. Must be a name in ``skimage.color.color_dict`` or RGB float\n        values between [0, 1].\n    image_alpha : float [0, 1], optional\n        Opacity of the image.\n    kind : string, one of {'overlay', 'avg'}\n        The kind of color image desired. 'overlay' cycles over defined colors\n        and overlays the colored labels over the original image. 'avg' replaces\n        each labeled segment with its average color, for a stained-class or\n        pastel painting appearance.\n    saturation : float [0, 1], optional\n        Parameter to control the saturation applied to the original image\n        between fully saturated (original RGB, `saturation=1`) and fully\n        unsaturated (grayscale, `saturation=0`). Only applies when\n        `kind='overlay'`.\n    channel_axis : int, optional\n        This parameter indicates which axis of the output array will correspond\n        to channels. If `image` is provided, this must also match the axis of\n        `image` that corresponds to channels.\n\n        .. versionadded:: 0.19\n            ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    result : ndarray of float, same shape as `image`\n        The result of blending a cycling colormap (`colors`) for each distinct\n        value in `label` with the image, at a certain alpha value.\n    "
    if image is not None:
        image = np.moveaxis(image, source=channel_axis, destination=-1)
    if kind == 'overlay':
        rgb = _label2rgb_overlay(label, image, colors, alpha, bg_label, bg_color, image_alpha, saturation)
    elif kind == 'avg':
        rgb = _label2rgb_avg(label, image, bg_label, bg_color)
    else:
        raise ValueError("`kind` must be either 'overlay' or 'avg'.")
    return np.moveaxis(rgb, source=-1, destination=channel_axis)

def _label2rgb_overlay(label, image=None, colors=None, alpha=0.3, bg_label=-1, bg_color=None, image_alpha=1, saturation=0):
    if False:
        while True:
            i = 10
    "Return an RGB image where color-coded labels are painted over the image.\n\n    Parameters\n    ----------\n    label : ndarray\n        Integer array of labels with the same shape as `image`.\n    image : ndarray, optional\n        Image used as underlay for labels. It should have the same shape as\n        `labels`, optionally with an additional RGB (channels) axis. If `image`\n        is an RGB image, it is converted to grayscale before coloring.\n    colors : list, optional\n        List of colors. If the number of labels exceeds the number of colors,\n        then the colors are cycled.\n    alpha : float [0, 1], optional\n        Opacity of colorized labels. Ignored if image is `None`.\n    bg_label : int, optional\n        Label that's treated as the background. If `bg_label` is specified and\n        `bg_color` is `None`, background is not painted by any colors.\n    bg_color : str or array, optional\n        Background color. Must be a name in ``skimage.color.color_dict`` or RGB float\n        values between [0, 1].\n    image_alpha : float [0, 1], optional\n        Opacity of the image.\n    saturation : float [0, 1], optional\n        Parameter to control the saturation applied to the original image\n        between fully saturated (original RGB, `saturation=1`) and fully\n        unsaturated (grayscale, `saturation=0`).\n\n    Returns\n    -------\n    result : ndarray of float, same shape as `image`\n        The result of blending a cycling colormap (`colors`) for each distinct\n        value in `label` with the image, at a certain alpha value.\n    "
    if not 0 <= saturation <= 1:
        warn(f'saturation must be in range [0, 1], got {saturation}')
    if colors is None:
        colors = DEFAULT_COLORS
    colors = [_rgb_vector(c) for c in colors]
    if image is None:
        image = np.zeros(label.shape + (3,), dtype=np.float64)
        alpha = 1
    else:
        if image.shape[:label.ndim] != label.shape or image.ndim > label.ndim + 1:
            raise ValueError('`image` and `label` must be the same shape')
        if image.ndim == label.ndim + 1 and image.shape[-1] != 3:
            raise ValueError('`image` must be RGB (image.shape[-1] must be 3).')
        if image.min() < 0:
            warn('Negative intensities in `image` are not supported')
        float_dtype = _supported_float_type(image.dtype)
        image = img_as_float(image).astype(float_dtype, copy=False)
        if image.ndim > label.ndim:
            hsv = rgb2hsv(image)
            hsv[..., 1] *= saturation
            image = hsv2rgb(hsv)
        elif image.ndim == label.ndim:
            image = gray2rgb(image)
        image = image * image_alpha + (1 - image_alpha)
    offset = min(label.min(), bg_label)
    if offset != 0:
        label = label - offset
        bg_label -= offset
    new_type = np.min_scalar_type(int(label.max()))
    if new_type == bool:
        new_type = np.uint8
    label = label.astype(new_type)
    (mapped_labels_flat, color_cycle) = _match_label_with_color(label, colors, bg_label, bg_color)
    if len(mapped_labels_flat) == 0:
        return image
    dense_labels = range(np.max(mapped_labels_flat) + 1)
    label_to_color = np.stack([c for (i, c) in zip(dense_labels, color_cycle)])
    mapped_labels = label
    mapped_labels.flat = mapped_labels_flat
    result = label_to_color[mapped_labels] * alpha + image * (1 - alpha)
    remove_background = 0 in mapped_labels_flat and bg_color is None
    if remove_background:
        result[label == bg_label] = image[label == bg_label]
    return result

def _label2rgb_avg(label_field, image, bg_label=0, bg_color=(0, 0, 0)):
    if False:
        while True:
            i = 10
    'Visualise each segment in `label_field` with its mean color in `image`.\n\n    Parameters\n    ----------\n    label_field : ndarray of int\n        A segmentation of an image.\n    image : array, shape ``label_field.shape + (3,)``\n        A color image of the same spatial shape as `label_field`.\n    bg_label : int, optional\n        A value in `label_field` to be treated as background.\n    bg_color : 3-tuple of int, optional\n        The color for the background label\n\n    Returns\n    -------\n    out : ndarray, same shape and type as `image`\n        The output visualization.\n    '
    out = np.zeros(label_field.shape + (3,), dtype=image.dtype)
    labels = np.unique(label_field)
    bg = labels == bg_label
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        color = image[mask].mean(axis=0)
        out[mask] = color
    return out