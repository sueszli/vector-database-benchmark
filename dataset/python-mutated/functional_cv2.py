import math
import numbers
from collections.abc import Iterable, Sequence
import numpy as np
import paddle
from paddle.utils import try_import
__all__ = []

def to_tensor(pic, data_format='CHW'):
    if False:
        for i in range(10):
            print('nop')
    "Converts a ``numpy.ndarray`` to paddle.Tensor.\n\n    See ``ToTensor`` for more details.\n\n    Args:\n        pic (np.ndarray): Image to be converted to tensor.\n        data_format (str, optional): Data format of output tensor, should be 'HWC' or\n            'CHW'. Default: 'CHW'.\n\n    Returns:\n        Tensor: Converted image.\n\n    "
    if data_format not in ['CHW', 'HWC']:
        raise ValueError(f'data_format should be CHW or HWC. Got {data_format}')
    if pic.ndim == 2:
        pic = pic[:, :, None]
    if data_format == 'CHW':
        img = paddle.to_tensor(pic.transpose((2, 0, 1)))
    else:
        img = paddle.to_tensor(pic)
    if paddle.base.data_feeder.convert_dtype(img.dtype) == 'uint8':
        return paddle.cast(img, np.float32) / 255.0
    else:
        return img

def resize(img, size, interpolation='bilinear'):
    if False:
        return 10
    '\n    Resizes the image to given size\n\n    Args:\n        input (np.ndarray): Image to be resized.\n        size (int|list|tuple): Target size of input data, with (height, width) shape.\n        interpolation (int|str, optional): Interpolation method. when use cv2 backend,\n            support method are as following:\n            - "nearest": cv2.INTER_NEAREST,\n            - "bilinear": cv2.INTER_LINEAR,\n            - "area": cv2.INTER_AREA,\n            - "bicubic": cv2.INTER_CUBIC,\n            - "lanczos": cv2.INTER_LANCZOS4\n\n    Returns:\n        np.array: Resized image.\n\n    '
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA, 'bicubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError(f'Got inappropriate size arg: {size}')
    (h, w) = img.shape[:2]
    if isinstance(size, int):
        if w <= h and w == size or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=_cv2_interp_from_str[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=_cv2_interp_from_str[interpolation])
    else:
        output = cv2.resize(img, dsize=(size[1], size[0]), interpolation=_cv2_interp_from_str[interpolation])
    if len(img.shape) == 3 and img.shape[2] == 1:
        return output[:, :, np.newaxis]
    else:
        return output

def pad(img, padding, fill=0, padding_mode='constant'):
    if False:
        while True:
            i = 10
    "\n    Pads the given numpy.array on all sides with specified padding mode and fill value.\n\n    Args:\n        img (np.array): Image to be padded.\n        padding (int|list|tuple): Padding on each border. If a single int is provided this\n            is used to pad all borders. If list/tuple of length 2 is provided this is the padding\n            on left/right and top/bottom respectively. If a list/tuple of length 4 is provided\n            this is the padding for the left, top, right and bottom borders\n            respectively.\n        fill (float, optional): Pixel fill value for constant fill. If a tuple of\n            length 3, it is used to fill R, G, B channels respectively.\n            This value is only used when the padding_mode is constant. Default: 0.\n        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default: 'constant'.\n\n            - constant: pads with a constant value, this value is specified with fill\n\n            - edge: pads with the last value on the edge of the image\n\n            - reflect: pads with reflection of image (without repeating the last value on the edge)\n\n                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode\n                       will result in [3, 2, 1, 2, 3, 4, 3, 2]\n\n            - symmetric: pads with reflection of image (repeating the last value on the edge)\n\n                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode\n                         will result in [2, 1, 1, 2, 3, 4, 4, 3]\n\n    Returns:\n        np.array: Padded image.\n\n    "
    cv2 = try_import('cv2')
    _cv2_pad_from_str = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE, 'reflect': cv2.BORDER_REFLECT_101, 'symmetric': cv2.BORDER_REFLECT}
    if not isinstance(padding, (numbers.Number, list, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, list, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        raise ValueError('Padding must be an int or a 2, or 4 element tuple, not a ' + f'{len(padding)} element tuple')
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], 'Padding mode should be either constant, edge, reflect or symmetric'
    if isinstance(padding, list):
        padding = tuple(padding)
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.copyMakeBorder(img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, borderType=_cv2_pad_from_str[padding_mode], value=fill)[:, :, np.newaxis]
    else:
        return cv2.copyMakeBorder(img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, borderType=_cv2_pad_from_str[padding_mode], value=fill)

def crop(img, top, left, height, width):
    if False:
        return 10
    'Crops the given image.\n\n    Args:\n        img (np.array): Image to be cropped. (0,0) denotes the top left\n            corner of the image.\n        top (int): Vertical component of the top left corner of the crop box.\n        left (int): Horizontal component of the top left corner of the crop box.\n        height (int): Height of the crop box.\n        width (int): Width of the crop box.\n\n    Returns:\n        np.array: Cropped image.\n\n    '
    return img[top:top + height, left:left + width, :]

def center_crop(img, output_size):
    if False:
        for i in range(10):
            print('nop')
    "Crops the given image and resize it to desired size.\n\n    Args:\n        img (np.array): Image to be cropped. (0,0) denotes the top left corner of the image.\n        output_size (sequence or int): (height, width) of the crop box. If int,\n            it is used for both directions\n        backend (str, optional): The image proccess backend type. Options are `pil`, `cv2`. Default: 'pil'.\n\n    Returns:\n        np.array: Cropped image.\n\n    "
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    (h, w) = img.shape[0:2]
    (th, tw) = output_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(img, i, j, th, tw)

def hflip(img):
    if False:
        return 10
    'Horizontally flips the given image.\n\n    Args:\n        img (np.array): Image to be flipped.\n\n    Returns:\n        np.array:  Horizontall flipped image.\n\n    '
    cv2 = try_import('cv2')
    return cv2.flip(img, 1)

def vflip(img):
    if False:
        print('Hello World!')
    'Vertically flips the given np.array.\n\n    Args:\n        img (np.array): Image to be flipped.\n\n    Returns:\n        np.array:  Vertically flipped image.\n\n    '
    cv2 = try_import('cv2')
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.flip(img, 0)[:, :, np.newaxis]
    else:
        return cv2.flip(img, 0)

def adjust_brightness(img, brightness_factor):
    if False:
        while True:
            i = 10
    'Adjusts brightness of an image.\n\n    Args:\n        img (np.array): Image to be adjusted.\n        brightness_factor (float):  How much to adjust the brightness. Can be\n            any non negative number. 0 gives a black image, 1 gives the\n            original image while 2 increases the brightness by a factor of 2.\n\n    Returns:\n        np.array: Brightness adjusted image.\n\n    '
    cv2 = try_import('cv2')
    table = np.array([i * brightness_factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)

def adjust_contrast(img, contrast_factor):
    if False:
        for i in range(10):
            print('nop')
    'Adjusts contrast of an image.\n\n    Args:\n        img (np.array): Image to be adjusted.\n        contrast_factor (float): How much to adjust the contrast. Can be any\n            non negative number. 0 gives a solid gray image, 1 gives the\n            original image while 2 increases the contrast by a factor of 2.\n\n    Returns:\n        np.array: Contrast adjusted image.\n\n    '
    cv2 = try_import('cv2')
    table = np.array([(i - 74) * contrast_factor + 74 for i in range(0, 256)]).clip(0, 255).astype('uint8')
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.LUT(img, table)[:, :, np.newaxis]
    else:
        return cv2.LUT(img, table)

def adjust_saturation(img, saturation_factor):
    if False:
        print('Hello World!')
    'Adjusts color saturation of an image.\n\n    Args:\n        img (np.array): Image to be adjusted.\n        saturation_factor (float):  How much to adjust the saturation. 0 will\n            give a black and white image, 1 will give the original image while\n            2 will enhance the saturation by a factor of 2.\n\n    Returns:\n        np.array: Saturation adjusted image.\n\n    '
    cv2 = try_import('cv2')
    dtype = img.dtype
    img = img.astype(np.float32)
    alpha = np.random.uniform(max(0, 1 - saturation_factor), 1 + saturation_factor)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img[..., np.newaxis]
    img = img * alpha + gray_img * (1 - alpha)
    return img.clip(0, 255).astype(dtype)

def adjust_hue(img, hue_factor):
    if False:
        for i in range(10):
            print('nop')
    'Adjusts hue of an image.\n\n    The image hue is adjusted by converting the image to HSV and\n    cyclically shifting the intensities in the hue channel (H).\n    The image is then converted back to original image mode.\n\n    `hue_factor` is the amount of shift in H channel and must be in the\n    interval `[-0.5, 0.5]`.\n\n    Args:\n        img (np.array): Image to be adjusted.\n        hue_factor (float):  How much to shift the hue channel. Should be in\n            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in\n            HSV space in positive and negative direction respectively.\n            0 means no shift. Therefore, both -0.5 and 0.5 will give an image\n            with complementary colors while 0 gives the original image.\n\n    Returns:\n        np.array: Hue adjusted image.\n\n    '
    cv2 = try_import('cv2')
    if not -0.5 <= hue_factor <= 0.5:
        raise ValueError(f'hue_factor:{hue_factor} is not in [-0.5, 0.5].')
    dtype = img.dtype
    img = img.astype(np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    (h, s, v) = cv2.split(hsv_img)
    alpha = np.random.uniform(hue_factor, hue_factor)
    h = h.astype(np.uint8)
    with np.errstate(over='ignore'):
        h += np.uint8(alpha * 255)
    hsv_img = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL).astype(dtype)

def affine(img, angle, translate, scale, shear, interpolation='nearest', fill=0, center=None):
    if False:
        i = 10
        return i + 15
    'Affine the image by matrix.\n\n    Args:\n        img (PIL.Image): Image to be affined.\n        translate (sequence or int): horizontal and vertical translations\n        scale (float): overall scale ratio\n        shear (sequence or float): shear angle value in degrees between -180 to 180, clockwise direction.\n            If a sequence is specified, the first value corresponds to a shear parallel to the x axis, while\n            the second value corresponds to a shear parallel to the y axis.\n        interpolation (int|str, optional): Interpolation method. If omitted, or if the\n            image has only one channel, it is set to cv2.INTER_NEAREST.\n            when use cv2 backend, support method are as following:\n            - "nearest": cv2.INTER_NEAREST,\n            - "bilinear": cv2.INTER_LINEAR,\n            - "bicubic": cv2.INTER_CUBIC\n        fill (3-tuple or int): RGB pixel fill value for area outside the affined image.\n            If int, it is used for all channels respectively.\n        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.\n            Default is the center of the image.\n\n    Returns:\n        np.array: Affined image.\n\n    '
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA, 'bicubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}
    (h, w) = img.shape[0:2]
    if isinstance(fill, int):
        fill = tuple([fill] * 3)
    if center is None:
        center = (w / 2.0, h / 2.0)
    M = np.ones([2, 3])
    R = cv2.getRotationMatrix2D(angle=angle, center=center, scale=scale)
    sx = math.tan(shear[0] * math.pi / 180)
    sy = math.tan(shear[1] * math.pi / 180)
    M[0] = R[0] + sy * R[1]
    M[1] = R[1] + sx * R[0]
    (tx, ty) = translate
    M[0, 2] = tx
    M[1, 2] = ty
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.warpAffine(img, M, dsize=(w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(img, M, dsize=(w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)

def rotate(img, angle, interpolation='nearest', expand=False, center=None, fill=0):
    if False:
        return 10
    'Rotates the image by angle.\n\n    Args:\n        img (np.array): Image to be rotated.\n        angle (float or int): In degrees degrees counter clockwise order.\n        interpolation (int|str, optional): Interpolation method. If omitted, or if the\n            image has only one channel, it is set to cv2.INTER_NEAREST.\n            when use cv2 backend, support method are as following:\n            - "nearest": cv2.INTER_NEAREST,\n            - "bilinear": cv2.INTER_LINEAR,\n            - "bicubic": cv2.INTER_CUBIC\n        expand (bool, optional): Optional expansion flag.\n            If true, expands the output image to make it large enough to hold the entire rotated image.\n            If false or omitted, make the output image the same size as the input image.\n            Note that the expand flag assumes rotation around the center and no translation.\n        center (2-tuple, optional): Optional center of rotation.\n            Origin is the upper left corner.\n            Default is the center of the image.\n        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.\n            If int, it is used for all channels respectively.\n\n    Returns:\n        np.array: Rotated image.\n\n    '
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA, 'bicubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}
    (h, w) = img.shape[0:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    if expand:

        def transform(x, y, matrix):
            if False:
                i = 10
                return i + 15
            (a, b, c, d, e, f) = matrix
            return (a * x + b * y + c, d * x + e * y + f)
        xx = []
        yy = []
        angle = -math.radians(angle)
        expand_matrix = [round(math.cos(angle), 15), round(math.sin(angle), 15), 0.0, round(-math.sin(angle), 15), round(math.cos(angle), 15), 0.0]
        post_trans = (0, 0)
        (expand_matrix[2], expand_matrix[5]) = transform(-center[0] - post_trans[0], -center[1] - post_trans[1], expand_matrix)
        expand_matrix[2] += center[0]
        expand_matrix[5] += center[1]
        for (x, y) in ((0, 0), (w, 0), (w, h), (0, h)):
            (x, y) = transform(x, y, expand_matrix)
            xx.append(x)
            yy.append(y)
        nw = math.ceil(max(xx)) - math.floor(min(xx))
        nh = math.ceil(max(yy)) - math.floor(min(yy))
        M[0, 2] += (nw - w) * 0.5
        M[1, 2] += (nh - h) * 0.5
        (w, h) = (int(nw), int(nh))
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.warpAffine(img, M, (w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)[:, :, np.newaxis]
    else:
        return cv2.warpAffine(img, M, (w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)

def perspective(img, startpoints, endpoints, interpolation='nearest', fill=0):
    if False:
        for i in range(10):
            print('nop')
    'Perspective the image.\n\n    Args:\n        img (np.array): Image to be perspectived.\n        startpoints (list[list[int]]): [top-left, top-right, bottom-right, bottom-left] of the original image,\n        endpoints (list[list[int]]): [top-left, top-right, bottom-right, bottom-left] of the transformed image.\n        interpolation (int|str, optional): Interpolation method. If omitted, or if the\n            image has only one channel, it is set to cv2.INTER_NEAREST.\n            when use cv2 backend, support method are as following:\n            - "nearest": cv2.INTER_NEAREST,\n            - "bilinear": cv2.INTER_LINEAR,\n            - "bicubic": cv2.INTER_CUBIC\n        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.\n            If int, it is used for all channels respectively.\n\n    Returns:\n        np.array: Perspectived image.\n\n    '
    cv2 = try_import('cv2')
    _cv2_interp_from_str = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA, 'bicubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}
    (h, w) = img.shape[0:2]
    startpoints = np.array(startpoints, dtype='float32')
    endpoints = np.array(endpoints, dtype='float32')
    matrix = cv2.getPerspectiveTransform(startpoints, endpoints)
    if len(img.shape) == 3 and img.shape[2] == 1:
        return cv2.warpPerspective(img, matrix, dsize=(w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)[:, :, np.newaxis]
    else:
        return cv2.warpPerspective(img, matrix, dsize=(w, h), flags=_cv2_interp_from_str[interpolation], borderValue=fill)

def to_grayscale(img, num_output_channels=1):
    if False:
        print('Hello World!')
    'Converts image to grayscale version of image.\n\n    Args:\n        img (np.array): Image to be converted to grayscale.\n\n    Returns:\n        np.array: Grayscale version of the image.\n            if num_output_channels = 1 : returned image is single channel\n\n            if num_output_channels = 3 : returned image is 3 channel with r = g = b\n\n    '
    cv2 = try_import('cv2')
    if num_output_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    elif num_output_channels == 3:
        img = np.broadcast_to(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis], img.shape)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')
    return img

def normalize(img, mean, std, data_format='CHW', to_rgb=False):
    if False:
        while True:
            i = 10
    "Normalizes a ndarray imge or image with mean and standard deviation.\n\n    Args:\n        img (np.array): input data to be normalized.\n        mean (list|tuple): Sequence of means for each channel.\n        std (list|tuple): Sequence of standard deviations for each channel.\n        data_format (str, optional): Data format of img, should be 'HWC' or\n            'CHW'. Default: 'CHW'.\n        to_rgb (bool, optional): Whether to convert to rgb. Default: False.\n\n    Returns:\n        np.array: Normalized mage.\n\n    "
    if data_format == 'CHW':
        mean = np.float32(np.array(mean).reshape(-1, 1, 1))
        std = np.float32(np.array(std).reshape(-1, 1, 1))
    else:
        mean = np.float32(np.array(mean).reshape(1, 1, -1))
        std = np.float32(np.array(std).reshape(1, 1, -1))
    if to_rgb:
        img = img[..., ::-1]
    img = (img - mean) / std
    return img

def erase(img, i, j, h, w, v, inplace=False):
    if False:
        for i in range(10):
            print('nop')
    'Erase the pixels of selected area in input image array with given value.\n\n    Args:\n         img (np.array): input image array, which shape is (H, W, C).\n         i (int): y coordinate of the top-left point of erased region.\n         j (int): x coordinate of the top-left point of erased region.\n         h (int): Height of the erased region.\n         w (int): Width of the erased region.\n         v (np.array): value used to replace the pixels in erased region.\n         inplace (bool, optional): Whether this transform is inplace. Default: False.\n\n     Returns:\n         np.array: Erased image.\n\n    '
    if not inplace:
        img = img.copy()
    img[i:i + h, j:j + w, ...] = v
    return img