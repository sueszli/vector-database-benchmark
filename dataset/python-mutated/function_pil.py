from typing import Sequence
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
import numpy as np
import numbers
import math
from math import cos, sin, tan

def _is_pil_image(img):
    if False:
        while True:
            i = 10
    return isinstance(img, Image.Image)

def _get_image_size(img):
    if False:
        for i in range(10):
            print('nop')
    if _is_pil_image(img):
        return img.size
    raise TypeError(f'Unexpected type {type(img)}')

def _get_image_num_channels(img):
    if False:
        i = 10
        return i + 15
    if _is_pil_image(img):
        return 1 if img.mode == 'L' else 3
    raise TypeError(f'Unexpected type {type(img)}')

def hflip(img):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function for horizontally flipping the given image.\n\n    Args::\n\n        [in] img(PIL Image.Image): Input image.\n\n    Example::\n        \n        img = Image.open(...)\n        img_ = transform.hflip(img)\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def vflip(img):
    if False:
        i = 10
        return i + 15
    '\n    Function for vertically flipping the given image.\n\n    Args::\n\n        [in] img(PIL Image.Image): Input image.\n\n    Example::\n        \n        img = Image.open(...)\n        img_ = transform.vflip(img)\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def adjust_brightness(img, brightness_factor):
    if False:
        print('Hello World!')
    '\n    Function for adjusting brightness of an RGB image.\n\n    Args::\n\n        [in] img (PIL Image.Image): Image to be adjusted.\n        [in] brightness_factor (float):  How much to adjust the brightness.\n             Can be any non negative number. 0 gives a black image, 1 gives the\n             original image while 2 increases the brightness by a factor of 2.\n\n    Returns::\n\n        [out] PIL Image.Image: Brightness adjusted image.\n\n    Example::\n        \n        img = Image.open(...)\n        img_ = transform.adjust_brightness(img, 0.5)\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img

def adjust_contrast(img, contrast_factor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function for adjusting contrast of an image.\n\n    Args::\n\n        [in] img (PIL Image.Image): Image to be adjusted.\n        [in] contrast_factor (float): How much to adjust the contrast.\n             Can be any non negative number. 0 gives a solid gray image,\n             1 gives the original image while 2 increases the contrast by a factor of 2.\n\n    Returns::\n\n        [out] PIL Image.Image: Contrast adjusted image.\n\n    Example::\n        \n        img = Image.open(...)\n        img_ = transform.adjust_contrast(img, 0.5)\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img

def adjust_saturation(img, saturation_factor):
    if False:
        print('Hello World!')
    '\n    Function for adjusting saturation of an image.\n\n    Args::\n\n        [in] img (PIL Image.Image): Image to be adjusted.\n        [in] saturation_factor (float):  How much to adjust the saturation.\n             0 will give a black and white image, 1 will give the original image\n             while 2 will enhance the saturation by a factor of 2.\n\n    Returns::\n\n        [out] PIL Image.Image: Saturation adjusted image.\n\n    Example::\n        \n        img = Image.open(...)\n        img_ = transform.adjust_saturation(img, 0.5)\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img

def adjust_hue(img, hue_factor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function for adjusting hue of an image.\n\n    The image hue is adjusted by converting the image to HSV and\n    cyclically shifting the intensities in the hue channel (H).\n    The image is then converted back to original image mode.\n\n    `hue_factor` is the amount of shift in H channel and must be in the\n    interval `[-0.5, 0.5]`.\n\n    See `Hue`_ for more details.\n\n    .. _Hue: https://en.wikipedia.org/wiki/Hue\n\n    Args::\n\n        [in] img (PIL Image.Image): Image to be adjusted.\n        [in] hue_factor (float):  How much to shift the hue channel.\n             Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of\n             hue channel in HSV space in positive and negative direction respectively.\n             0 means no shift. Therefore, both -0.5 and 0.5 will give an image\n             with complementary colors while 0 gives the original image.\n\n    Returns::\n\n        [out] PIL Image.Image: Saturation adjusted image.\n\n    Example::\n        \n        img = Image.open(...)\n        img_ = transform.adjust_hue(img, 0.1)\n    '
    if not -0.5 <= hue_factor <= 0.5:
        raise ValueError(f'hue_factor ({hue_factor}) is not in [-0.5, 0.5].')
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img
    (h, s, v) = img.convert('HSV').split()
    np_h = np.array(h, dtype=np.uint8)
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')
    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img

def adjust_gamma(img, gamma, gain=1):
    if False:
        while True:
            i = 10
    '\n    Function for performing gamma correction on an image.\n\n    Also known as Power Law Transform. Intensities in RGB mode are adjusted\n    based on the following equation:\n\n    .. math::\n        I_{\text{out}} = 255 \times \text{gain} \times \\left(\x0crac{I_{\text{in}}}{255}\right)^{\\gamma}\n\n    See `Gamma Correction`_ for more details.\n\n    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction\n\n    Args::\n\n        [in] img (PIL Image.Image): Image to be adjusted.\n        [in] gamma (float): Non negative real number, same as :math:`\\gamma` in the equation.\n             gamma larger than 1 make the shadows darker,\n             while gamma smaller than 1 make dark regions lighter.\n        [in] gain (float): The constant multiplier.\n\n    Returns::\n\n        [out] PIL Image.Image: Gamma adjusted image.\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')
    input_mode = img.mode
    img = img.convert('RGB')
    gamma_map = [int((255 + 1 - 0.001) * gain * pow(ele / 255.0, gamma)) for ele in range(256)] * 3
    img = img.point(gamma_map)
    img = img.convert(input_mode)
    return img

def crop(img, top, left, height, width):
    if False:
        print('Hello World!')
    '\n    Function for cropping image.\n\n    Args::\n\n        [in] img(PIL Image.Image): Input image.\n        [in] top(int): the top boundary of the cropping box.\n        [in] left(int): the left boundary of the cropping box.\n        [in] height(int): height of the cropping box.\n        [in] width(int): width of the cropping box.\n\n    Returns::\n\n        [out] PIL Image.Image: Cropped image.\n\n    Example::\n        \n        img = Image.open(...)\n        img_ = transform.crop(img, 10, 10, 100, 100)\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    return img.crop((left, top, left + width, top + height))

def resize(img, size, interpolation=Image.BILINEAR):
    if False:
        i = 10
        return i + 15
    '\n    Function for resizing the input image to the given size.\n\n    Args::\n\n        [in] img(PIL Image.Image): Input image.\n        [in] size(sequence or int): Desired output size. If size is a sequence like\n             (h, w), the output size will be matched to this. If size is an int,\n             the smaller edge of the image will be matched to this number maintaining\n             the aspect ratio. If a tuple or list of length 1 is provided, it is\n             interpreted as a single int.\n        [in] interpolation(int, optional): type of interpolation. default: PIL.Image.BILINEAR\n\n    Returns::\n\n        [out] PIL Image.Image: Resized image.\n\n    Example::\n        \n        img = Image.open(...)\n        img_ = transform.resize(img, (100, 100))\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError(f'Got inappropriate size arg: {size}')
    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        (w, h) = img.size
        if w <= h and w == size or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

def gray(img, num_output_channels):
    if False:
        print('Hello World!')
    '\n    Function for converting PIL image of any mode (RGB, HSV, LAB, etc) to grayscale version of image.\n\n    Args::\n\n        [in] img(PIL Image.Image): Input image.\n        [in] num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.\n\n    Returns::\n\n        [out] PIL Image: Grayscale version of the image.\n              if num_output_channels = 1 : returned image is single channel\n              if num_output_channels = 3 : returned image is 3 channel with r = g = b\n    '
    if not _is_pil_image(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')
    return img

def _get_perspective_coeffs(startpoints, endpoints):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.\n\n    In Perspective Transform each pixel (x, y) in the orignal image gets transformed as,\n     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )\n\n    Args:\n        List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image,\n        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed\n                   image\n    Returns:\n        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.\n    '
    matrix = []
    for (p1, p2) in zip(endpoints, startpoints):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A = np.array(matrix, dtype='float')
    B = np.array(startpoints, dtype='float').reshape(8)
    res = np.linalg.lstsq(A, B, rcond=-1)[0]
    return res.tolist()

def perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC):
    if False:
        while True:
            i = 10
    'Perform perspective transform of the given PIL Image.\n\n    Args:\n        img (PIL Image): Image to be transformed.\n        startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image\n        endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image\n        interpolation: Default- Image.BICUBIC\n    Returns:\n        PIL Image:  Perspectively transformed Image.\n    '
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation)

def resized_crop(img, top, left, height, width, size, interpolation=Image.BILINEAR):
    if False:
        for i in range(10):
            print('nop')
    'Crop the given PIL Image and resize it to desired size.\n\n    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.\n\n    Args:\n        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.\n        top (int): Vertical component of the top left corner of the crop box.\n        left (int): Horizontal component of the top left corner of the crop box.\n        height (int): Height of the crop box.\n        width (int): Width of the crop box.\n        size (sequence or int): Desired output size. Same semantics as ``resize``.\n        interpolation (int, optional): Desired interpolation. Default is\n            ``PIL.Image.BILINEAR``.\n    Returns:\n        PIL Image: Cropped image.\n    '
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img

def center_crop(img, output_size):
    if False:
        for i in range(10):
            print('nop')
    'Crop the given PIL Image and resize it to desired size.\n\n        Args:\n            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.\n            output_size (sequence or int): (height, width) of the crop box. If int,\n                it is used for both directions\n        Returns:\n            PIL Image: Cropped image.\n        '
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    (image_width, image_height) = img.size
    (crop_height, crop_width) = output_size
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(img, crop_top, crop_left, crop_height, crop_width)

def five_crop(img, size):
    if False:
        print('Hello World!')
    'Crop the given PIL Image into four corners and the central crop.\n\n    .. Note::\n        This transform returns a tuple of images and there may be a\n        mismatch in the number of inputs and targets your ``Dataset`` returns.\n\n    Args:\n       size (sequence or int): Desired output size of the crop. If size is an\n           int instead of sequence like (h, w), a square crop (size, size) is\n           made.\n\n    Returns:\n       tuple: tuple (tl, tr, bl, br, center)\n                Corresponding top left, top right, bottom left, bottom right and center crop.\n    '
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, 'Please provide only two dimensions (h, w) for size.'
    (image_width, image_height) = img.size
    (crop_height, crop_width) = size
    if crop_width > image_width or crop_height > image_height:
        msg = 'Requested crop size {} is bigger than input size {}'
        raise ValueError(msg.format(size, (image_height, image_width)))
    tl = img.crop((0, 0, crop_width, crop_height))
    tr = img.crop((image_width - crop_width, 0, image_width, crop_height))
    bl = img.crop((0, image_height - crop_height, crop_width, image_height))
    br = img.crop((image_width - crop_width, image_height - crop_height, image_width, image_height))
    center = center_crop(img, (crop_height, crop_width))
    return (tl, tr, bl, br, center)

def ten_crop(img, size, vertical_flip=False):
    if False:
        return 10
    'Crop the given PIL Image into four corners and the central crop plus the\n        flipped version of these (horizontal flipping is used by default).\n\n    .. Note::\n        This transform returns a tuple of images and there may be a\n        mismatch in the number of inputs and targets your ``Dataset`` returns.\n\n    Args:\n       size (sequence or int): Desired output size of the crop. If size is an\n            int instead of sequence like (h, w), a square crop (size, size) is\n            made.\n       vertical_flip (bool): Use vertical flipping instead of horizontal\n\n    Returns:\n       tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)\n                Corresponding top left, top right, bottom left, bottom right and center crop\n                and same for the flipped image.\n    '
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, 'Please provide only two dimensions (h, w) for size.'
    first_five = five_crop(img, size)
    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)
    second_five = five_crop(img, size)
    return first_five + second_five

def rotate(img, angle, resample=False, expand=False, center=None, fill=None):
    if False:
        while True:
            i = 10
    'Rotate the image by angle.\n\n\n    Args:\n        img (PIL Image): PIL Image to be rotated.\n        angle (float or int): In degrees degrees counter clockwise order.\n        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):\n            An optional resampling filter. See `filters`_ for more information.\n            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.\n        expand (bool, optional): Optional expansion flag.\n            If true, expands the output image to make it large enough to hold the entire rotated image.\n            If false or omitted, make the output image the same size as the input image.\n            Note that the expand flag assumes rotation around the center and no translation.\n        center (2-tuple, optional): Optional center of rotation.\n            Origin is the upper left corner.\n            Default is the center of the image.\n        fill (n-tuple or int or float): Pixel fill value for area outside the rotated\n            image. If int or float, the value is used for all bands respectively.\n            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.\n\n    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters\n\n    '

    def parse_fill(fill, num_bands):
        if False:
            while True:
                i = 10
        if PILLOW_VERSION < '5.2.0':
            if fill is None:
                return {}
            else:
                msg = 'The option to fill background area of the rotated image, requires pillow>=5.2.0'
                raise RuntimeError(msg)
        if fill is None:
            fill = 0
        if isinstance(fill, (int, float)) and num_bands > 1:
            fill = tuple([fill] * num_bands)
        if not isinstance(fill, (int, float)) and len(fill) != num_bands:
            msg = "The number of elements in 'fill' does not match the number of bands of the image ({} != {})"
            raise ValueError(msg.format(len(fill), num_bands))
        return {'fillcolor': fill}
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    opts = parse_fill(fill, len(img.getbands()))
    return img.rotate(angle, resample, expand, center, **opts)

def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    if False:
        return 10
    if isinstance(shear, numbers.Number):
        shear = [shear, 0]
    if not isinstance(shear, (tuple, list)) and len(shear) == 2:
        raise ValueError('Shear should be a single value or a tuple/list containing ' + 'two values. Got {}'.format(shear))
    rot = math.radians(angle)
    (sx, sy) = [math.radians(s) for s in shear]
    (cx, cy) = center
    (tx, ty) = translate
    a = cos(rot - sy) / cos(sy)
    b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
    c = sin(rot - sy) / cos(sy)
    d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)
    M = [d, -b, 0, -c, a, 0]
    M = [x / scale for x in M]
    M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
    M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)
    M[2] += cx
    M[5] += cy
    return M

def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    if False:
        return 10
    'Apply affine transformation on the image keeping image center invariant\n\n    Args:\n        img (PIL Image): PIL Image to be rotated.\n        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.\n        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)\n        scale (float): overall scale\n        shear (float or tuple or list): shear angle value in degrees between -180 to 180, clockwise direction.\n        If a tuple of list is specified, the first value corresponds to a shear parallel to the x axis, while\n        the second value corresponds to a shear parallel to the y axis.\n        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):\n            An optional resampling filter.\n            See `filters`_ for more information.\n            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.\n        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)\n    '
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    assert isinstance(translate, (tuple, list)) and len(translate) == 2, 'Argument translate should be a list or tuple of length 2'
    assert scale > 0.0, 'Argument scale should be positive'
    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {'fillcolor': fillcolor} if PILLOW_VERSION[0] >= '5' else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)