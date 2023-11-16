"""Functions for converting between color spaces.

The "central" color space in this module is RGB, more specifically the linear
sRGB color space using D65 as a white-point [1]_.  This represents a
standard monitor (w/o gamma correction). For a good FAQ on color spaces see
[2]_.

The API consists of functions to convert to and from RGB as defined above, as
well as a generic function to convert to and from any supported color space
(which is done through RGB in most cases).


Supported color spaces
----------------------
* RGB : Red Green Blue.
        Here the sRGB standard [1]_.
* HSV : Hue, Saturation, Value.
        Uniquely defined when related to sRGB [3]_.
* RGB CIE : Red Green Blue.
        The original RGB CIE standard from 1931 [4]_. Primary colors are 700 nm
        (red), 546.1 nm (blue) and 435.8 nm (green).
* XYZ CIE : XYZ
        Derived from the RGB CIE color space. Chosen such that
        ``x == y == z == 1/3`` at the whitepoint, and all color matching
        functions are greater than zero everywhere.
* LAB CIE : Lightness, a, b
        Colorspace derived from XYZ CIE that is intended to be more
        perceptually uniform
* LUV CIE : Lightness, u, v
        Colorspace derived from XYZ CIE that is intended to be more
        perceptually uniform
* LCH CIE : Lightness, Chroma, Hue
        Defined in terms of LAB CIE.  C and H are the polar representation of
        a and b.  The polar angle C is defined to be on ``(0, 2*pi)``

:author: Nicolas Pinto (rgb2hsv)
:author: Ralf Gommers (hsv2rgb)
:author: Travis Oliphant (XYZ and RGB CIE functions)
:author: Matt Terry (lab2lch)
:author: Alex Izvorski (yuv2rgb, rgb2yuv and related)

:license: modified BSD

References
----------
.. [1] Official specification of sRGB, IEC 61966-2-1:1999.
.. [2] http://www.poynton.com/ColorFAQ.html
.. [3] https://en.wikipedia.org/wiki/HSL_and_HSV
.. [4] https://en.wikipedia.org/wiki/CIE_1931_color_space
"""
from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import _supported_float_type, channel_as_last_axis, identity, reshape_nd, slice_at_axis, deprecate_func
from ..util import dtype, dtype_limits
try:
    from numpy import AxisError
except ImportError:
    from numpy.exceptions import AxisError

def convert_colorspace(arr, fromspace, tospace, *, channel_axis=-1):
    if False:
        return 10
    'Convert an image array to a new color space.\n\n    Valid color spaces are:\n        \'RGB\', \'HSV\', \'RGB CIE\', \'XYZ\', \'YUV\', \'YIQ\', \'YPbPr\', \'YCbCr\', \'YDbDr\'\n\n    Parameters\n    ----------\n    arr : (..., C=3, ...) array_like\n        The image to convert. By default, the final dimension denotes\n        channels.\n    fromspace : str\n        The color space to convert from. Can be specified in lower case.\n    tospace : str\n        The color space to convert to. Can be specified in lower case.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The converted image. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If fromspace is not a valid color space\n    ValueError\n        If tospace is not a valid color space\n\n    Notes\n    -----\n    Conversion is performed through the "central" RGB color space,\n    i.e. conversion from XYZ to HSV is implemented as ``XYZ -> RGB -> HSV``\n    instead of directly.\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> img = data.astronaut()\n    >>> img_hsv = convert_colorspace(img, \'RGB\', \'HSV\')\n    '
    fromdict = {'rgb': identity, 'hsv': hsv2rgb, 'rgb cie': rgbcie2rgb, 'xyz': xyz2rgb, 'yuv': yuv2rgb, 'yiq': yiq2rgb, 'ypbpr': ypbpr2rgb, 'ycbcr': ycbcr2rgb, 'ydbdr': ydbdr2rgb}
    todict = {'rgb': identity, 'hsv': rgb2hsv, 'rgb cie': rgb2rgbcie, 'xyz': rgb2xyz, 'yuv': rgb2yuv, 'yiq': rgb2yiq, 'ypbpr': rgb2ypbpr, 'ycbcr': rgb2ycbcr, 'ydbdr': rgb2ydbdr}
    fromspace = fromspace.lower()
    tospace = tospace.lower()
    if fromspace not in fromdict:
        msg = f'`fromspace` has to be one of {fromdict.keys()}'
        raise ValueError(msg)
    if tospace not in todict:
        msg = f'`tospace` has to be one of {todict.keys()}'
        raise ValueError(msg)
    return todict[tospace](fromdict[fromspace](arr, channel_axis=channel_axis), channel_axis=channel_axis)

def _prepare_colorarray(arr, force_copy=False, *, channel_axis=-1):
    if False:
        i = 10
        return i + 15
    'Check the shape of the array and convert it to\n    floating point representation.\n    '
    arr = np.asanyarray(arr)
    if arr.shape[channel_axis] != 3:
        msg = f'the input array must have size 3 along `channel_axis`, got {arr.shape}'
        raise ValueError(msg)
    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == np.float32:
        _func = dtype.img_as_float32
    else:
        _func = dtype.img_as_float64
    return _func(arr, force_copy=force_copy)

def _validate_channel_axis(channel_axis, ndim):
    if False:
        print('Hello World!')
    if not isinstance(channel_axis, int):
        raise TypeError('channel_axis must be an integer')
    if channel_axis < -ndim or channel_axis >= ndim:
        raise AxisError('channel_axis exceeds array dimensions')

def rgba2rgb(rgba, background=(1, 1, 1), *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'RGBA to RGB conversion using alpha blending [1]_.\n\n    Parameters\n    ----------\n    rgba : (..., C=4, ...) array_like\n        The image in RGBA format. By default, the final dimension denotes\n        channels.\n    background : array_like\n        The color of the background to blend the image with (3 floats\n        between 0 to 1 - the RGB value of the background).\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgba` is not at least 2D with shape (..., 4, ...).\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending\n\n    Examples\n    --------\n    >>> from skimage import color\n    >>> from skimage import data\n    >>> img_rgba = data.logo()\n    >>> img_rgb = color.rgba2rgb(img_rgba)\n    '
    arr = np.asanyarray(rgba)
    _validate_channel_axis(channel_axis, arr.ndim)
    channel_axis = channel_axis % arr.ndim
    if arr.shape[channel_axis] != 4:
        msg = f'the input array must have size 4 along `channel_axis`, got {arr.shape}'
        raise ValueError(msg)
    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == np.float32:
        arr = dtype.img_as_float32(arr)
    else:
        arr = dtype.img_as_float64(arr)
    background = np.ravel(background).astype(arr.dtype)
    if len(background) != 3:
        raise ValueError(f'background must be an array-like containing 3 RGB values. Got {len(background)} items')
    if np.any(background < 0) or np.any(background > 1):
        raise ValueError('background RGB values must be floats between 0 and 1.')
    background = reshape_nd(background, arr.ndim, channel_axis)
    alpha = arr[slice_at_axis(slice(3, 4), axis=channel_axis)]
    channels = arr[slice_at_axis(slice(3), axis=channel_axis)]
    out = np.clip((1 - alpha) * background + alpha * channels, a_min=0, a_max=1)
    return out

@channel_as_last_axis()
def rgb2hsv(rgb, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'RGB to HSV color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in HSV format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    Conversion between RGB and HSV color spaces results in some loss of\n    precision, due to integer arithmetic and rounding [1]_.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV\n\n    Examples\n    --------\n    >>> from skimage import color\n    >>> from skimage import data\n    >>> img = data.astronaut()\n    >>> img_hsv = color.rgb2hsv(img)\n    '
    input_is_one_pixel = rgb.ndim == 1
    if input_is_one_pixel:
        rgb = rgb[np.newaxis, ...]
    arr = _prepare_colorarray(rgb, channel_axis=-1)
    out = np.empty_like(arr)
    out_v = arr.max(-1)
    delta = arr.ptp(-1)
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.0] = 0.0
    idx = arr[..., 0] == out_v
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    idx = arr[..., 1] == out_v
    out[idx, 0] = 2.0 + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    idx = arr[..., 2] == out_v
    out[idx, 0] = 4.0 + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = out[..., 0] / 6.0 % 1.0
    out_h[delta == 0.0] = 0.0
    np.seterr(**old_settings)
    out[..., 0] = out_h
    out[..., 1] = out_s
    out[..., 2] = out_v
    out[np.isnan(out)] = 0
    if input_is_one_pixel:
        out = np.squeeze(out, axis=0)
    return out

@channel_as_last_axis()
def hsv2rgb(hsv, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'HSV to RGB color space conversion.\n\n    Parameters\n    ----------\n    hsv : (..., C=3, ...) array_like\n        The image in HSV format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `hsv` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    Conversion between RGB and HSV color spaces results in some loss of\n    precision, due to integer arithmetic and rounding [1]_.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> img = data.astronaut()\n    >>> img_hsv = rgb2hsv(img)\n    >>> img_rgb = hsv2rgb(img_hsv)\n    '
    arr = _prepare_colorarray(hsv, channel_axis=-1)
    hi = np.floor(arr[..., 0] * 6)
    f = arr[..., 0] * 6 - hi
    p = arr[..., 2] * (1 - arr[..., 1])
    q = arr[..., 2] * (1 - f * arr[..., 1])
    t = arr[..., 2] * (1 - (1 - f) * arr[..., 1])
    v = arr[..., 2]
    hi = np.stack([hi, hi, hi], axis=-1).astype(np.uint8) % 6
    out = np.choose(hi, np.stack([np.stack((v, t, p), axis=-1), np.stack((q, v, p), axis=-1), np.stack((p, v, t), axis=-1), np.stack((p, q, v), axis=-1), np.stack((t, p, v), axis=-1), np.stack((v, p, q), axis=-1)]))
    return out
cie_primaries = np.array([700, 546.1, 435.8])
sb_primaries = np.array([1.0 / 155, 1.0 / 190, 1.0 / 225]) * 100000.0
xyz_from_rgb = np.array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])
rgb_from_xyz = linalg.inv(xyz_from_rgb)
xyz_from_rgbcie = np.array([[0.49, 0.31, 0.2], [0.17697, 0.8124, 0.01063], [0.0, 0.01, 0.99]]) / 0.17697
rgbcie_from_xyz = linalg.inv(xyz_from_rgbcie)
rgbcie_from_rgb = rgbcie_from_xyz @ xyz_from_rgb
rgb_from_rgbcie = rgb_from_xyz @ xyz_from_rgbcie
gray_from_rgb = np.array([[0.2125, 0.7154, 0.0721], [0, 0, 0], [0, 0, 0]])
yuv_from_rgb = np.array([[0.299, 0.587, 0.114], [-0.14714119, -0.28886916, 0.43601035], [0.61497538, -0.51496512, -0.10001026]])
rgb_from_yuv = linalg.inv(yuv_from_rgb)
yiq_from_rgb = np.array([[0.299, 0.587, 0.114], [0.59590059, -0.27455667, -0.32134392], [0.21153661, -0.52273617, 0.31119955]])
rgb_from_yiq = linalg.inv(yiq_from_rgb)
ypbpr_from_rgb = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
rgb_from_ypbpr = linalg.inv(ypbpr_from_rgb)
ycbcr_from_rgb = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]])
rgb_from_ycbcr = linalg.inv(ycbcr_from_rgb)
ydbdr_from_rgb = np.array([[0.299, 0.587, 0.114], [-0.45, -0.883, 1.333], [-1.333, 1.116, 0.217]])
rgb_from_ydbdr = linalg.inv(ydbdr_from_rgb)
lab_ref_white = np.array([0.95047, 1.0, 1.08883])
_illuminants = {'A': {'2': (1.098466069456375, 1, 0.3558228003436005), '10': (1.111420406956693, 1, 0.3519978321919493), 'R': (1.098466069456375, 1, 0.3558228003436005)}, 'B': {'2': (0.9909274480248003, 1, 0.8531327322886154), '10': (0.9917777147717607, 1, 0.8434930535866175), 'R': (0.9909274480248003, 1, 0.8531327322886154)}, 'C': {'2': (0.980705971659919, 1, 1.1822494939271255), '10': (0.9728569189782166, 1, 1.1614480488951577), 'R': (0.980705971659919, 1, 1.1822494939271255)}, 'D50': {'2': (0.9642119944211994, 1, 0.8251882845188288), '10': (0.9672062750333777, 1, 0.8142801513128616), 'R': (0.9639501491621826, 1, 0.8241280285499208)}, 'D55': {'2': (0.956797052643698, 1, 0.9214805860173273), '10': (0.9579665682254781, 1, 0.9092525159847462), 'R': (0.9565317453467969, 1, 0.9202554587037198)}, 'D65': {'2': (0.95047, 1.0, 1.08883), '10': (0.94809667673716, 1, 1.0730513595166162), 'R': (0.9532057125493769, 1, 1.0853843816469158)}, 'D75': {'2': (0.9497220898840717, 1, 1.226393520724154), '10': (0.9441713925645873, 1, 1.2064272211720228), 'R': (0.9497220898840717, 1, 1.226393520724154)}, 'E': {'2': (1.0, 1.0, 1.0), '10': (1.0, 1.0, 1.0), 'R': (1.0, 1.0, 1.0)}}

def xyz_tristimulus_values(*, illuminant, observer, dtype=float):
    if False:
        print('Hello World!')
    'Get the CIE XYZ tristimulus values.\n\n    Given an illuminant and observer, this function returns the CIE XYZ tristimulus\n    values [2]_ scaled such that :math:`Y = 1`.\n\n    Parameters\n    ----------\n    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}\n        The name of the illuminant (the function is NOT case sensitive).\n    observer : {"2", "10", "R"}\n        One of: 2-degree observer, 10-degree observer, or \'R\' observer as in\n        R function ``grDevices::convertColor`` [3]_.\n    dtype: dtype, optional\n        Output data type.\n\n    Returns\n    -------\n    values : array\n        Array with 3 elements :math:`X, Y, Z` containing the CIE XYZ tristimulus values\n        of the given illuminant.\n\n    Raises\n    ------\n    ValueError\n        If either the illuminant or the observer angle are not supported or\n        unknown.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants\n    .. [2] https://en.wikipedia.org/wiki/CIE_1931_color_space#Meaning_of_X,_Y_and_Z\n    .. [3] https://www.rdocumentation.org/packages/grDevices/versions/3.6.2/topics/convertColor\n\n    Notes\n    -----\n    The CIE XYZ tristimulus values are calculated from :math:`x, y` [1]_, using the\n    formula\n\n    .. math:: X = x / y\n\n    .. math:: Y = 1\n\n    .. math:: Z = (1 - x - y) / y\n\n    The only exception is the illuminant "D65" with aperture angle 2° for\n    backward-compatibility reasons.\n\n    Examples\n    --------\n    Get the CIE XYZ tristimulus values for a "D65" illuminant for a 10 degree field of\n    view\n\n    >>> xyz_tristimulus_values(illuminant="D65", observer="10")\n    array([0.94809668, 1.        , 1.07305136])\n    '
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return np.asarray(_illuminants[illuminant][observer], dtype=dtype)
    except KeyError:
        raise ValueError(f'Unknown illuminant/observer combination (`{illuminant}`, `{observer}`)')

@deprecate_func(hint='Use `skimage.color.xyz_tristimulus_values` instead.', deprecated_version='0.21', removed_version='0.23')
def get_xyz_coords(illuminant, observer, dtype=float):
    if False:
        i = 10
        return i + 15
    'Get the XYZ coordinates of the given illuminant and observer [1]_.\n\n    Parameters\n    ----------\n    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional\n        The name of the illuminant (the function is NOT case sensitive).\n    observer : {"2", "10", "R"}, optional\n        One of: 2-degree observer, 10-degree observer, or \'R\' observer as in\n        R function grDevices::convertColor.\n    dtype: dtype, optional\n        Output data type.\n\n    Returns\n    -------\n    out : array\n        Array with 3 elements containing the XYZ coordinates of the given\n        illuminant.\n\n    Raises\n    ------\n    ValueError\n        If either the illuminant or the observer angle are not supported or\n        unknown.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant\n    '
    return xyz_tristimulus_values(illuminant=illuminant, observer=observer, dtype=dtype)
rgb_from_hed = np.array([[0.65, 0.7, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
hed_from_rgb = linalg.inv(rgb_from_hed)
rgb_from_hdx = np.array([[0.65, 0.704, 0.286], [0.268, 0.57, 0.776], [0.0, 0.0, 0.0]])
rgb_from_hdx[2, :] = np.cross(rgb_from_hdx[0, :], rgb_from_hdx[1, :])
hdx_from_rgb = linalg.inv(rgb_from_hdx)
rgb_from_fgx = np.array([[0.46420921, 0.83008335, 0.30827187], [0.94705542, 0.25373821, 0.19650764], [0.0, 0.0, 0.0]])
rgb_from_fgx[2, :] = np.cross(rgb_from_fgx[0, :], rgb_from_fgx[1, :])
fgx_from_rgb = linalg.inv(rgb_from_fgx)
rgb_from_bex = np.array([[0.834750233, 0.513556283, 0.196330403], [0.092789, 0.954111, 0.283111], [0.0, 0.0, 0.0]])
rgb_from_bex[2, :] = np.cross(rgb_from_bex[0, :], rgb_from_bex[1, :])
bex_from_rgb = linalg.inv(rgb_from_bex)
rgb_from_rbd = np.array([[0.21393921, 0.85112669, 0.47794022], [0.74890292, 0.60624161, 0.26731082], [0.268, 0.57, 0.776]])
rbd_from_rgb = linalg.inv(rgb_from_rbd)
rgb_from_gdx = np.array([[0.98003, 0.144316, 0.133146], [0.268, 0.57, 0.776], [0.0, 0.0, 0.0]])
rgb_from_gdx[2, :] = np.cross(rgb_from_gdx[0, :], rgb_from_gdx[1, :])
gdx_from_rgb = linalg.inv(rgb_from_gdx)
rgb_from_hax = np.array([[0.65, 0.704, 0.286], [0.2743, 0.6796, 0.6803], [0.0, 0.0, 0.0]])
rgb_from_hax[2, :] = np.cross(rgb_from_hax[0, :], rgb_from_hax[1, :])
hax_from_rgb = linalg.inv(rgb_from_hax)
rgb_from_bro = np.array([[0.853033, 0.508733, 0.112656], [0.09289875, 0.8662008, 0.49098468], [0.10732849, 0.36765403, 0.9237484]])
bro_from_rgb = linalg.inv(rgb_from_bro)
rgb_from_bpx = np.array([[0.7995107, 0.5913521, 0.10528667], [0.09997159, 0.73738605, 0.6680326], [0.0, 0.0, 0.0]])
rgb_from_bpx[2, :] = np.cross(rgb_from_bpx[0, :], rgb_from_bpx[1, :])
bpx_from_rgb = linalg.inv(rgb_from_bpx)
rgb_from_ahx = np.array([[0.874622, 0.457711, 0.158256], [0.552556, 0.7544, 0.353744], [0.0, 0.0, 0.0]])
rgb_from_ahx[2, :] = np.cross(rgb_from_ahx[0, :], rgb_from_ahx[1, :])
ahx_from_rgb = linalg.inv(rgb_from_ahx)
rgb_from_hpx = np.array([[0.644211, 0.716556, 0.266844], [0.175411, 0.972178, 0.154589], [0.0, 0.0, 0.0]])
rgb_from_hpx[2, :] = np.cross(rgb_from_hpx[0, :], rgb_from_hpx[1, :])
hpx_from_rgb = linalg.inv(rgb_from_hpx)

def _convert(matrix, arr):
    if False:
        print('Hello World!')
    'Do the color space conversion.\n\n    Parameters\n    ----------\n    matrix : array_like\n        The 3x3 matrix to use.\n    arr : (..., C=3, ...) array_like\n        The input array. By default, the final dimension denotes\n        channels.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The converted array. Same dimensions as input.\n    '
    arr = _prepare_colorarray(arr)
    return arr @ matrix.T.astype(arr.dtype)

@channel_as_last_axis()
def xyz2rgb(xyz, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'XYZ to RGB color space conversion.\n\n    Parameters\n    ----------\n    xyz : (..., C=3, ...) array_like\n        The image in XYZ format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `xyz` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    The CIE XYZ color space is derived from the CIE RGB color space. Note\n    however that this function converts to sRGB.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2xyz, xyz2rgb\n    >>> img = data.astronaut()\n    >>> img_xyz = rgb2xyz(img)\n    >>> img_rgb = xyz2rgb(img_xyz)\n    '
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    np.clip(arr, 0, 1, out=arr)
    return arr

@channel_as_last_axis()
def rgb2xyz(rgb, *, channel_axis=-1):
    if False:
        return 10
    'RGB to XYZ color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in XYZ format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    The CIE XYZ color space is derived from the CIE RGB color space. Note\n    however that this function converts from sRGB.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> img = data.astronaut()\n    >>> img_xyz = rgb2xyz(img)\n    '
    arr = _prepare_colorarray(rgb, channel_axis=-1).copy()
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return arr @ xyz_from_rgb.T.astype(arr.dtype)

@channel_as_last_axis()
def rgb2rgbcie(rgb, *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'RGB to RGB CIE color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB CIE format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2rgbcie\n    >>> img = data.astronaut()\n    >>> img_rgbcie = rgb2rgbcie(img)\n    '
    return _convert(rgbcie_from_rgb, rgb)

@channel_as_last_axis()
def rgbcie2rgb(rgbcie, *, channel_axis=-1):
    if False:
        return 10
    'RGB CIE to RGB color space conversion.\n\n    Parameters\n    ----------\n    rgbcie : (..., C=3, ...) array_like\n        The image in RGB CIE format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgbcie` is not at least 2-D with shape (..., C=3, ...).\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2rgbcie, rgbcie2rgb\n    >>> img = data.astronaut()\n    >>> img_rgbcie = rgb2rgbcie(img)\n    >>> img_rgb = rgbcie2rgb(img_rgbcie)\n    '
    return _convert(rgb_from_rgbcie, rgbcie)

@channel_as_last_axis(multichannel_output=False)
def rgb2gray(rgb, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'Compute luminance of an RGB image.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n\n    Returns\n    -------\n    out : ndarray\n        The luminance image - an array which is the same size as the input\n        array, but with the channel dimension removed.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    The weights used in this conversion are calibrated for contemporary\n    CRT phosphors::\n\n        Y = 0.2125 R + 0.7154 G + 0.0721 B\n\n    If there is an alpha channel present, it is ignored.\n\n    References\n    ----------\n    .. [1] http://poynton.ca/PDFs/ColorFAQ.pdf\n\n    Examples\n    --------\n    >>> from skimage.color import rgb2gray\n    >>> from skimage import data\n    >>> img = data.astronaut()\n    >>> img_gray = rgb2gray(img)\n    '
    rgb = _prepare_colorarray(rgb)
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=rgb.dtype)
    return rgb @ coeffs

def gray2rgba(image, alpha=None, *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'Create a RGBA representation of a gray-level image.\n\n    Parameters\n    ----------\n    image : array_like\n        Input image.\n    alpha : array_like, optional\n        Alpha channel of the output image. It may be a scalar or an\n        array that can be broadcast to ``image``. If not specified it is\n        set to the maximum limit corresponding to the ``image`` dtype.\n    channel_axis : int, optional\n        This parameter indicates which axis of the output array will correspond\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    rgba : ndarray\n        RGBA image. A new dimension of length 4 is added to input\n        image shape.\n    '
    arr = np.asarray(image)
    (alpha_min, alpha_max) = dtype_limits(arr, clip_negative=False)
    if alpha is None:
        alpha = alpha_max
    if not np.can_cast(alpha, arr.dtype):
        warn(f'alpha cannot be safely cast to image dtype {arr.dtype.name}', stacklevel=2)
    if np.isscalar(alpha):
        alpha = np.full(arr.shape, alpha, dtype=arr.dtype)
    elif alpha.shape != arr.shape:
        raise ValueError('alpha.shape must match image.shape')
    rgba = np.stack((arr,) * 3 + (alpha,), axis=channel_axis)
    return rgba

def gray2rgb(image, *, channel_axis=-1):
    if False:
        return 10
    'Create an RGB representation of a gray-level image.\n\n    Parameters\n    ----------\n    image : array_like\n        Input image.\n    channel_axis : int, optional\n        This parameter indicates which axis of the output array will correspond\n        to channels.\n\n    Returns\n    -------\n    rgb : (..., C=3, ...) ndarray\n        RGB image. A new dimension of length 3 is added to input image.\n\n    Notes\n    -----\n    If the input is a 1-dimensional image of shape ``(M,)``, the output\n    will be shape ``(M, C=3)``.\n    '
    return np.stack(3 * (image,), axis=channel_axis)

@channel_as_last_axis()
def xyz2lab(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    if False:
        return 10
    'XYZ to CIE-LAB color space conversion.\n\n    Parameters\n    ----------\n    xyz : (..., C=3, ...) array_like\n        The image in XYZ format. By default, the final dimension denotes\n        channels.\n    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional\n        The name of the illuminant (the function is NOT case sensitive).\n    observer : {"2", "10", "R"}, optional\n        One of: 2-degree observer, 10-degree observer, or \'R\' observer as in\n        R function grDevices::convertColor.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in CIE-LAB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `xyz` is not at least 2-D with shape (..., C=3, ...).\n    ValueError\n        If either the illuminant or the observer angle is unsupported or\n        unknown.\n\n    Notes\n    -----\n    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values\n    x_ref=95.047, y_ref=100., z_ref=108.883. See function\n    :func:`~.xyz_tristimulus_values` for a list of supported illuminants.\n\n    References\n    ----------\n    .. [1] http://www.easyrgb.com/en/math.php\n    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2xyz, xyz2lab\n    >>> img = data.astronaut()\n    >>> img_xyz = rgb2xyz(img)\n    >>> img_lab = xyz2lab(img_xyz)\n    '
    arr = _prepare_colorarray(xyz, channel_axis=-1)
    xyz_ref_white = xyz_tristimulus_values(illuminant=illuminant, observer=observer, dtype=arr.dtype)
    arr = arr / xyz_ref_white
    mask = arr > 0.008856
    arr[mask] = np.cbrt(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16.0 / 116.0
    (x, y, z) = (arr[..., 0], arr[..., 1], arr[..., 2])
    L = 116.0 * y - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)

@channel_as_last_axis()
def lab2xyz(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'Convert image in CIE-LAB to XYZ color space.\n\n    Parameters\n    ----------\n    lab : (..., C=3, ...) array_like\n        The input image in CIE-LAB color space.\n        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB\n        channels.\n        The L* values range from 0 to 100;\n        the a* and b* values range from -128 to 127.\n    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional\n        The name of the illuminant (the function is NOT case sensitive).\n    observer : {"2", "10", "R"}, optional\n        The aperture angle of the observer.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in XYZ color space, of same shape as input.\n\n    Raises\n    ------\n    ValueError\n        If `lab` is not at least 2-D with shape (..., C=3, ...).\n    ValueError\n        If either the illuminant or the observer angle are not supported or\n        unknown.\n    UserWarning\n        If any of the pixels are invalid (Z < 0).\n\n    Notes\n    -----\n    The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and\n    z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of\n    supported illuminants.\n\n    See Also\n    --------\n    xyz2lab\n\n    References\n    ----------\n    .. [1] http://www.easyrgb.com/en/math.php\n    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space\n    '
    (xyz, n_invalid) = _lab2xyz(lab, illuminant, observer)
    if n_invalid != 0:
        warn(f'Conversion from CIE-LAB to XYZ color space resulted in {n_invalid} negative Z values that have been clipped to zero', stacklevel=3)
    return xyz

def _lab2xyz(lab, illuminant, observer):
    if False:
        print('Hello World!')
    'Convert CIE-LAB to XYZ color space.\n\n    Internal function for :func:`~.lab2xyz` and others. In addition to the\n    converted image, return the number of invalid pixels in the Z channel for\n    correct warning propagation.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in XYZ format. Same dimensions as input.\n    n_invalid : int\n        Number of invalid pixels in the Z channel after conversion.\n    '
    arr = _prepare_colorarray(lab, channel_axis=-1).copy()
    (L, a, b) = (arr[..., 0], arr[..., 1], arr[..., 2])
    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = y - b / 200.0
    invalid = np.atleast_1d(z < 0).nonzero()
    n_invalid = invalid[0].size
    if n_invalid != 0:
        if z.ndim > 0:
            z[invalid] = 0
        else:
            z = 0
    out = np.stack([x, y, z], axis=-1)
    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.0)
    out[~mask] = (out[~mask] - 16.0 / 116.0) / 7.787
    xyz_ref_white = xyz_tristimulus_values(illuminant=illuminant, observer=observer)
    out *= xyz_ref_white
    return (out, n_invalid)

@channel_as_last_axis()
def rgb2lab(rgb, illuminant='D65', observer='2', *, channel_axis=-1):
    if False:
        return 10
    'Conversion from the sRGB color space (IEC 61966-2-1:1999)\n    to the CIE Lab colorspace under the given illuminant and observer.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional\n        The name of the illuminant (the function is NOT case sensitive).\n    observer : {"2", "10", "R"}, optional\n        The aperture angle of the observer.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in Lab format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    RGB is a device-dependent color space so, if you use this function, be\n    sure that the image you are analyzing has been mapped to the sRGB color\n    space.\n\n    This function uses rgb2xyz and xyz2lab.\n    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values\n    x_ref=95.047, y_ref=100., z_ref=108.883. See function\n    :func:`~.xyz_tristimulus_values` for a list of supported illuminants.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant\n    '
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)

@channel_as_last_axis()
def lab2rgb(lab, illuminant='D65', observer='2', *, channel_axis=-1):
    if False:
        i = 10
        return i + 15
    'Convert image in CIE-LAB to sRGB color space.\n\n    Parameters\n    ----------\n    lab : (..., C=3, ...) array_like\n        The input image in CIE-LAB color space.\n        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB\n        channels.\n        The L* values range from 0 to 100;\n        the a* and b* values range from -128 to 127.\n    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional\n        The name of the illuminant (the function is NOT case sensitive).\n    observer : {"2", "10", "R"}, optional\n        The aperture angle of the observer.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in sRGB color space, of same shape as input.\n\n    Raises\n    ------\n    ValueError\n        If `lab` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    This function uses :func:`~.lab2xyz` and :func:`~.xyz2rgb`.\n    The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and\n    z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of\n    supported illuminants.\n\n    See Also\n    --------\n    rgb2lab\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant\n    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space\n    '
    (xyz, n_invalid) = _lab2xyz(lab, illuminant, observer)
    if n_invalid != 0:
        warn(f'Conversion from CIE-LAB, via XYZ to sRGB color space resulted in {n_invalid} negative Z values that have been clipped to zero', stacklevel=3)
    return xyz2rgb(xyz)

@channel_as_last_axis()
def xyz2luv(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    if False:
        return 10
    'XYZ to CIE-Luv color space conversion.\n\n    Parameters\n    ----------\n    xyz : (..., C=3, ...) array_like\n        The image in XYZ format. By default, the final dimension denotes\n        channels.\n    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional\n        The name of the illuminant (the function is NOT case sensitive).\n    observer : {"2", "10", "R"}, optional\n        The aperture angle of the observer.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in CIE-Luv format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `xyz` is not at least 2-D with shape (..., C=3, ...).\n    ValueError\n        If either the illuminant or the observer angle are not supported or\n        unknown.\n\n    Notes\n    -----\n    By default XYZ conversion weights use observer=2A. Reference whitepoint\n    for D65 Illuminant, with XYZ tristimulus values of ``(95.047, 100.,\n    108.883)``. See function :func:`~.xyz_tristimulus_values` for a list of supported\n    illuminants.\n\n    References\n    ----------\n    .. [1] http://www.easyrgb.com/en/math.php\n    .. [2] https://en.wikipedia.org/wiki/CIELUV\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2xyz, xyz2luv\n    >>> img = data.astronaut()\n    >>> img_xyz = rgb2xyz(img)\n    >>> img_luv = xyz2luv(img_xyz)\n    '
    input_is_one_pixel = xyz.ndim == 1
    if input_is_one_pixel:
        xyz = xyz[np.newaxis, ...]
    arr = _prepare_colorarray(xyz, channel_axis=-1)
    (x, y, z) = (arr[..., 0], arr[..., 1], arr[..., 2])
    eps = np.finfo(float).eps
    xyz_ref_white = np.array(xyz_tristimulus_values(illuminant=illuminant, observer=observer))
    L = y / xyz_ref_white[1]
    mask = L > 0.008856
    L[mask] = 116.0 * np.cbrt(L[mask]) - 16.0
    L[~mask] = 903.3 * L[~mask]
    u0 = 4 * xyz_ref_white[0] / ([1, 15, 3] @ xyz_ref_white)
    v0 = 9 * xyz_ref_white[1] / ([1, 15, 3] @ xyz_ref_white)

    def fu(X, Y, Z):
        if False:
            print('Hello World!')
        return 4.0 * X / (X + 15.0 * Y + 3.0 * Z + eps)

    def fv(X, Y, Z):
        if False:
            for i in range(10):
                print('nop')
        return 9.0 * Y / (X + 15.0 * Y + 3.0 * Z + eps)
    u = 13.0 * L * (fu(x, y, z) - u0)
    v = 13.0 * L * (fv(x, y, z) - v0)
    out = np.stack([L, u, v], axis=-1)
    if input_is_one_pixel:
        out = np.squeeze(out, axis=0)
    return out

@channel_as_last_axis()
def luv2xyz(luv, illuminant='D65', observer='2', *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'CIE-Luv to XYZ color space conversion.\n\n    Parameters\n    ----------\n    luv : (..., C=3, ...) array_like\n        The image in CIE-Luv format. By default, the final dimension denotes\n        channels.\n    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional\n        The name of the illuminant (the function is NOT case sensitive).\n    observer : {"2", "10", "R"}, optional\n        The aperture angle of the observer.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in XYZ format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `luv` is not at least 2-D with shape (..., C=3, ...).\n    ValueError\n        If either the illuminant or the observer angle are not supported or\n        unknown.\n\n    Notes\n    -----\n    XYZ conversion weights use observer=2A. Reference whitepoint for D65\n    Illuminant, with XYZ tristimulus values of ``(95.047, 100., 108.883)``. See\n    function :func:`~.xyz_tristimulus_values` for a list of supported illuminants.\n\n    References\n    ----------\n    .. [1] http://www.easyrgb.com/en/math.php\n    .. [2] https://en.wikipedia.org/wiki/CIELUV\n    '
    arr = _prepare_colorarray(luv, channel_axis=-1).copy()
    (L, u, v) = (arr[..., 0], arr[..., 1], arr[..., 2])
    eps = np.finfo(float).eps
    y = L.copy()
    mask = y > 7.999625
    y[mask] = np.power((y[mask] + 16.0) / 116.0, 3.0)
    y[~mask] = y[~mask] / 903.3
    xyz_ref_white = xyz_tristimulus_values(illuminant=illuminant, observer=observer)
    y *= xyz_ref_white[1]
    uv_weights = np.array([1, 15, 3])
    u0 = 4 * xyz_ref_white[0] / (uv_weights @ xyz_ref_white)
    v0 = 9 * xyz_ref_white[1] / (uv_weights @ xyz_ref_white)
    a = u0 + u / (13.0 * L + eps)
    b = v0 + v / (13.0 * L + eps)
    c = 3 * y * (5 * b - 3)
    z = ((a - 4) * c - 15 * a * b * y) / (12 * b)
    x = -(c / b + 3.0 * z)
    return np.concatenate([q[..., np.newaxis] for q in [x, y, z]], axis=-1)

@channel_as_last_axis()
def rgb2luv(rgb, *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'RGB to CIE-Luv color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in CIE Luv format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    This function uses rgb2xyz and xyz2luv.\n\n    References\n    ----------\n    .. [1] http://www.easyrgb.com/en/math.php\n    .. [2] https://en.wikipedia.org/wiki/CIELUV\n    '
    return xyz2luv(rgb2xyz(rgb))

@channel_as_last_axis()
def luv2rgb(luv, *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'Luv to RGB color space conversion.\n\n    Parameters\n    ----------\n    luv : (..., C=3, ...) array_like\n        The image in CIE Luv format. By default, the final dimension denotes\n        channels.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `luv` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    This function uses luv2xyz and xyz2rgb.\n    '
    return xyz2rgb(luv2xyz(luv))

@channel_as_last_axis()
def rgb2hed(rgb, *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in HED format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    References\n    ----------\n    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical\n           staining by color deconvolution.," Analytical and quantitative\n           cytology and histology / the International Academy of Cytology [and]\n           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2hed\n    >>> ihc = data.immunohistochemistry()\n    >>> ihc_hed = rgb2hed(ihc)\n    '
    return separate_stains(rgb, hed_from_rgb)

@channel_as_last_axis()
def hed2rgb(hed, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'Haematoxylin-Eosin-DAB (HED) to RGB color space conversion.\n\n    Parameters\n    ----------\n    hed : (..., C=3, ...) array_like\n        The image in the HED color space. By default, the final dimension\n        denotes channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `hed` is not at least 2-D with shape (..., C=3, ...).\n\n    References\n    ----------\n    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical\n           staining by color deconvolution.," Analytical and quantitative\n           cytology and histology / the International Academy of Cytology [and]\n           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2hed, hed2rgb\n    >>> ihc = data.immunohistochemistry()\n    >>> ihc_hed = rgb2hed(ihc)\n    >>> ihc_rgb = hed2rgb(ihc_hed)\n    '
    return combine_stains(hed, rgb_from_hed)

@channel_as_last_axis()
def separate_stains(rgb, conv_matrix, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'RGB to stain color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    conv_matrix: ndarray\n        The stain separation matrix as described by G. Landini [1]_.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in stain color space. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    Stain separation matrices available in the ``color`` module and their\n    respective colorspace:\n\n    * ``hed_from_rgb``: Hematoxylin + Eosin + DAB\n    * ``hdx_from_rgb``: Hematoxylin + DAB\n    * ``fgx_from_rgb``: Feulgen + Light Green\n    * ``bex_from_rgb``: Giemsa stain : Methyl Blue + Eosin\n    * ``rbd_from_rgb``: FastRed + FastBlue +  DAB\n    * ``gdx_from_rgb``: Methyl Green + DAB\n    * ``hax_from_rgb``: Hematoxylin + AEC\n    * ``bro_from_rgb``: Blue matrix Anilline Blue + Red matrix Azocarmine                        + Orange matrix Orange-G\n    * ``bpx_from_rgb``: Methyl Blue + Ponceau Fuchsin\n    * ``ahx_from_rgb``: Alcian Blue + Hematoxylin\n    * ``hpx_from_rgb``: Hematoxylin + PAS\n\n    This implementation borrows some ideas from DIPlib [2]_, e.g. the\n    compensation using a small value to avoid log artifacts when\n    calculating the Beer-Lambert law.\n\n    References\n    ----------\n    .. [1] https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html\n    .. [2] https://github.com/DIPlib/diplib/\n    .. [3] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical\n           staining by color deconvolution,” Anal. Quant. Cytol. Histol., vol.\n           23, no. 4, pp. 291–299, Aug. 2001.\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import separate_stains, hdx_from_rgb\n    >>> ihc = data.immunohistochemistry()\n    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)\n    '
    rgb = _prepare_colorarray(rgb, force_copy=True, channel_axis=-1)
    np.maximum(rgb, 1e-06, out=rgb)
    log_adjust = np.log(1e-06)
    stains = np.log(rgb) / log_adjust @ conv_matrix
    np.maximum(stains, 0, out=stains)
    return stains

@channel_as_last_axis()
def combine_stains(stains, conv_matrix, *, channel_axis=-1):
    if False:
        i = 10
        return i + 15
    'Stain to RGB color space conversion.\n\n    Parameters\n    ----------\n    stains : (..., C=3, ...) array_like\n        The image in stain color space. By default, the final dimension denotes\n        channels.\n    conv_matrix: ndarray\n        The stain separation matrix as described by G. Landini [1]_.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `stains` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    Stain combination matrices available in the ``color`` module and their\n    respective colorspace:\n\n    * ``rgb_from_hed``: Hematoxylin + Eosin + DAB\n    * ``rgb_from_hdx``: Hematoxylin + DAB\n    * ``rgb_from_fgx``: Feulgen + Light Green\n    * ``rgb_from_bex``: Giemsa stain : Methyl Blue + Eosin\n    * ``rgb_from_rbd``: FastRed + FastBlue +  DAB\n    * ``rgb_from_gdx``: Methyl Green + DAB\n    * ``rgb_from_hax``: Hematoxylin + AEC\n    * ``rgb_from_bro``: Blue matrix Anilline Blue + Red matrix Azocarmine                        + Orange matrix Orange-G\n    * ``rgb_from_bpx``: Methyl Blue + Ponceau Fuchsin\n    * ``rgb_from_ahx``: Alcian Blue + Hematoxylin\n    * ``rgb_from_hpx``: Hematoxylin + PAS\n\n    References\n    ----------\n    .. [1] https://web.archive.org/web/20160624145052/http://www.mecourse.com/landinig/software/cdeconv/cdeconv.html\n    .. [2] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical\n           staining by color deconvolution,” Anal. Quant. Cytol. Histol., vol.\n           23, no. 4, pp. 291–299, Aug. 2001.\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import (separate_stains, combine_stains,\n    ...                            hdx_from_rgb, rgb_from_hdx)\n    >>> ihc = data.immunohistochemistry()\n    >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)\n    >>> ihc_rgb = combine_stains(ihc_hdx, rgb_from_hdx)\n    '
    stains = _prepare_colorarray(stains, channel_axis=-1)
    log_adjust = -np.log(1e-06)
    log_rgb = -(stains * log_adjust) @ conv_matrix
    rgb = np.exp(log_rgb)
    return np.clip(rgb, a_min=0, a_max=1)

@channel_as_last_axis()
def lab2lch(lab, *, channel_axis=-1):
    if False:
        return 10
    'Convert image in CIE-LAB to CIE-LCh color space.\n\n    CIE-LCh is the cylindrical representation of the CIE-LAB (Cartesian) color\n    space.\n\n    Parameters\n    ----------\n    lab : (..., C=3, ...) array_like\n        The input image in CIE-LAB color space.\n        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB\n        channels.\n        The L* values range from 0 to 100;\n        the a* and b* values range from -128 to 127.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in CIE-LCh color space, of same shape as input.\n\n    Raises\n    ------\n    ValueError\n        If `lab` does not have at least 3 channels (i.e., L*, a*, and b*).\n\n    Notes\n    -----\n    The h channel (i.e., hue) is expressed as an angle in range ``(0, 2*pi)``.\n\n    See Also\n    --------\n    lch2lab\n\n    References\n    ----------\n    .. [1] http://www.easyrgb.com/en/math.php\n    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space\n    .. [3] https://en.wikipedia.org/wiki/HCL_color_space\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2lab, lab2lch\n    >>> img = data.astronaut()\n    >>> img_lab = rgb2lab(img)\n    >>> img_lch = lab2lch(img_lab)\n    '
    lch = _prepare_lab_array(lab)
    (a, b) = (lch[..., 1], lch[..., 2])
    (lch[..., 1], lch[..., 2]) = _cart2polar_2pi(a, b)
    return lch

def _cart2polar_2pi(x, y):
    if False:
        for i in range(10):
            print('nop')
    'convert cartesian coordinates to polar (uses non-standard theta range!)\n\n    NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than usual ``(-pi, +pi)``\n    '
    (r, t) = (np.hypot(x, y), np.arctan2(y, x))
    t += np.where(t < 0.0, 2 * np.pi, 0)
    return (r, t)

@channel_as_last_axis()
def lch2lab(lch, *, channel_axis=-1):
    if False:
        print('Hello World!')
    'Convert image in CIE-LCh to CIE-LAB color space.\n\n    CIE-LCh is the cylindrical representation of the CIE-LAB (Cartesian) color\n    space.\n\n    Parameters\n    ----------\n    lch : (..., C=3, ...) array_like\n        The input image in CIE-LCh color space.\n        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB\n        channels.\n        The L* values range from 0 to 100;\n        the C values range from 0 to 100;\n        the h values range from 0 to ``2*pi``.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in CIE-LAB format, of same shape as input.\n\n    Raises\n    ------\n    ValueError\n        If `lch` does not have at least 3 channels (i.e., L*, C, and h).\n\n    Notes\n    -----\n    The h channel (i.e., hue) is expressed as an angle in range ``(0, 2*pi)``.\n\n    See Also\n    --------\n    lab2lch\n\n    References\n    ----------\n    .. [1] http://www.easyrgb.com/en/math.php\n    .. [2] https://en.wikipedia.org/wiki/HCL_color_space\n    .. [3] https://en.wikipedia.org/wiki/CIELAB_color_space\n\n    Examples\n    --------\n    >>> from skimage import data\n    >>> from skimage.color import rgb2lab, lch2lab, lab2lch\n    >>> img = data.astronaut()\n    >>> img_lab = rgb2lab(img)\n    >>> img_lch = lab2lch(img_lab)\n    >>> img_lab2 = lch2lab(img_lch)\n    '
    lch = _prepare_lab_array(lch)
    (c, h) = (lch[..., 1], lch[..., 2])
    (lch[..., 1], lch[..., 2]) = (c * np.cos(h), c * np.sin(h))
    return lch

def _prepare_lab_array(arr, force_copy=True):
    if False:
        print('Hello World!')
    'Ensure input for lab2lch and lch2lab is well-formed.\n\n    Input array must be in floating point and have at least 3 elements in the\n    last dimension. Returns a new array by default.\n    '
    arr = np.asarray(arr)
    shape = arr.shape
    if shape[-1] < 3:
        raise ValueError('Input image has less than 3 channels.')
    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == np.float32:
        _func = dtype.img_as_float32
    else:
        _func = dtype.img_as_float64
    return _func(arr, force_copy=force_copy)

@channel_as_last_axis()
def rgb2yuv(rgb, *, channel_axis=-1):
    if False:
        print('Hello World!')
    'RGB to YUV color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in YUV format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    Y is between 0 and 1.  Use YCbCr instead of YUV for the color space\n    commonly used by video codecs, where Y ranges from 16 to 235.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/YUV\n    '
    return _convert(yuv_from_rgb, rgb)

@channel_as_last_axis()
def rgb2yiq(rgb, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'RGB to YIQ color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in YIQ format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n    '
    return _convert(yiq_from_rgb, rgb)

@channel_as_last_axis()
def rgb2ypbpr(rgb, *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'RGB to YPbPr color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in YPbPr format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/YPbPr\n    '
    return _convert(ypbpr_from_rgb, rgb)

@channel_as_last_axis()
def rgb2ycbcr(rgb, *, channel_axis=-1):
    if False:
        i = 10
        return i + 15
    'RGB to YCbCr color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in YCbCr format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    Y is between 16 and 235. This is the color space commonly used by video\n    codecs; it is sometimes incorrectly called "YUV".\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/YCbCr\n    '
    arr = _convert(ycbcr_from_rgb, rgb)
    arr[..., 0] += 16
    arr[..., 1] += 128
    arr[..., 2] += 128
    return arr

@channel_as_last_axis()
def rgb2ydbdr(rgb, *, channel_axis=-1):
    if False:
        i = 10
        return i + 15
    'RGB to YDbDr color space conversion.\n\n    Parameters\n    ----------\n    rgb : (..., C=3, ...) array_like\n        The image in RGB format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in YDbDr format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `rgb` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    This is the color space commonly used by video codecs. It is also the\n    reversible color transform in JPEG2000.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/YDbDr\n    '
    arr = _convert(ydbdr_from_rgb, rgb)
    return arr

@channel_as_last_axis()
def yuv2rgb(yuv, *, channel_axis=-1):
    if False:
        i = 10
        return i + 15
    'YUV to RGB color space conversion.\n\n    Parameters\n    ----------\n    yuv : (..., C=3, ...) array_like\n        The image in YUV format. By default, the final dimension denotes\n        channels.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `yuv` is not at least 2-D with shape (..., C=3, ...).\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/YUV\n    '
    return _convert(rgb_from_yuv, yuv)

@channel_as_last_axis()
def yiq2rgb(yiq, *, channel_axis=-1):
    if False:
        print('Hello World!')
    'YIQ to RGB color space conversion.\n\n    Parameters\n    ----------\n    yiq : (..., C=3, ...) array_like\n        The image in YIQ format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `yiq` is not at least 2-D with shape (..., C=3, ...).\n    '
    return _convert(rgb_from_yiq, yiq)

@channel_as_last_axis()
def ypbpr2rgb(ypbpr, *, channel_axis=-1):
    if False:
        while True:
            i = 10
    'YPbPr to RGB color space conversion.\n\n    Parameters\n    ----------\n    ypbpr : (..., C=3, ...) array_like\n        The image in YPbPr format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `ypbpr` is not at least 2-D with shape (..., C=3, ...).\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/YPbPr\n    '
    return _convert(rgb_from_ypbpr, ypbpr)

@channel_as_last_axis()
def ycbcr2rgb(ycbcr, *, channel_axis=-1):
    if False:
        print('Hello World!')
    'YCbCr to RGB color space conversion.\n\n    Parameters\n    ----------\n    ycbcr : (..., C=3, ...) array_like\n        The image in YCbCr format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `ycbcr` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    Y is between 16 and 235. This is the color space commonly used by video\n    codecs; it is sometimes incorrectly called "YUV".\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/YCbCr\n    '
    arr = ycbcr.copy()
    arr[..., 0] -= 16
    arr[..., 1] -= 128
    arr[..., 2] -= 128
    return _convert(rgb_from_ycbcr, arr)

@channel_as_last_axis()
def ydbdr2rgb(ydbdr, *, channel_axis=-1):
    if False:
        for i in range(10):
            print('nop')
    'YDbDr to RGB color space conversion.\n\n    Parameters\n    ----------\n    ydbdr : (..., C=3, ...) array_like\n        The image in YDbDr format. By default, the final dimension denotes\n        channels.\n    channel_axis : int, optional\n        This parameter indicates which axis of the array corresponds to\n        channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (..., C=3, ...) ndarray\n        The image in RGB format. Same dimensions as input.\n\n    Raises\n    ------\n    ValueError\n        If `ydbdr` is not at least 2-D with shape (..., C=3, ...).\n\n    Notes\n    -----\n    This is the color space commonly used by video codecs, also called the\n    reversible color transform in JPEG2000.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/YDbDr\n    '
    return _convert(rgb_from_ydbdr, ydbdr)