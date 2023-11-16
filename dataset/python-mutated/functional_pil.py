import numbers
from collections.abc import Iterable, Sequence
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import paddle
try:
    _pil_interp_from_str = {'nearest': Image.Resampling.NEAREST, 'bilinear': Image.Resampling.BILINEAR, 'bicubic': Image.Resampling.BICUBIC, 'box': Image.Resampling.BOX, 'lanczos': Image.Resampling.LANCZOS, 'hamming': Image.Resampling.HAMMING}
except:
    _pil_interp_from_str = {'nearest': Image.NEAREST, 'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC, 'box': Image.BOX, 'lanczos': Image.LANCZOS, 'hamming': Image.HAMMING}
__all__ = []

def to_tensor(pic, data_format='CHW'):
    if False:
        return 10
    "Converts a ``PIL.Image`` to paddle.Tensor.\n\n    See ``ToTensor`` for more details.\n\n    Args:\n        pic (PIL.Image): Image to be converted to tensor.\n        data_format (str, optional): Data format of output tensor, should be 'HWC' or\n            'CHW'. Default: 'CHW'.\n\n    Returns:\n        Tensor: Converted image.\n\n    "
    if data_format not in ['CHW', 'HWC']:
        raise ValueError(f'data_format should be CHW or HWC. Got {data_format}')
    if pic.mode == 'I':
        img = paddle.to_tensor(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = paddle.to_tensor(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'F':
        img = paddle.to_tensor(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * paddle.to_tensor(np.array(pic, np.uint8, copy=False))
    else:
        img = paddle.to_tensor(np.array(pic, copy=False))
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    dtype = paddle.base.data_feeder.convert_dtype(img.dtype)
    if dtype == 'uint8':
        img = paddle.cast(img, np.float32) / 255.0
    img = img.reshape([pic.size[1], pic.size[0], nchannel])
    if data_format == 'CHW':
        img = img.transpose([2, 0, 1])
    return img

def resize(img, size, interpolation='bilinear'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Resizes the image to given size\n\n    Args:\n        input (PIL.Image): Image to be resized.\n        size (int|list|tuple): Target size of input data, with (height, width) shape.\n        interpolation (int|str, optional): Interpolation method. when use pil backend,\n            support method are as following:\n            - "nearest": Image.NEAREST,\n            - "bilinear": Image.BILINEAR,\n            - "bicubic": Image.BICUBIC,\n            - "box": Image.BOX,\n            - "lanczos": Image.LANCZOS,\n            - "hamming": Image.HAMMING\n\n    Returns:\n        PIL.Image: Resized image.\n\n    '
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError(f'Got inappropriate size arg: {size}')
    if isinstance(size, int):
        (w, h) = img.size
        if w <= h and w == size or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), _pil_interp_from_str[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), _pil_interp_from_str[interpolation])
    else:
        return img.resize(size[::-1], _pil_interp_from_str[interpolation])

def pad(img, padding, fill=0, padding_mode='constant'):
    if False:
        return 10
    "\n    Pads the given PIL.Image on all sides with specified padding mode and fill value.\n\n    Args:\n        img (PIL.Image): Image to be padded.\n        padding (int|list|tuple): Padding on each border. If a single int is provided this\n            is used to pad all borders. If list/tuple of length 2 is provided this is the padding\n            on left/right and top/bottom respectively. If a list/tuple of length 4 is provided\n            this is the padding for the left, top, right and bottom borders\n            respectively.\n        fill (float, optional): Pixel fill value for constant fill. If a tuple of\n            length 3, it is used to fill R, G, B channels respectively.\n            This value is only used when the padding_mode is constant. Default: 0.\n        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default: 'constant'.\n\n            - constant: pads with a constant value, this value is specified with fill\n\n            - edge: pads with the last value on the edge of the image\n\n            - reflect: pads with reflection of image (without repeating the last value on the edge)\n\n                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode\n                       will result in [3, 2, 1, 2, 3, 4, 3, 2]\n\n            - symmetric: pads with reflection of image (repeating the last value on the edge)\n\n                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode\n                         will result in [2, 1, 1, 2, 3, 4, 4, 3]\n\n    Returns:\n        PIL.Image: Padded image.\n\n    "
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
    if padding_mode == 'constant':
        if img.mode == 'P':
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, fill=fill)
            image.putpalette(palette)
            return image
        return ImageOps.expand(img, border=padding, fill=fill)
    else:
        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img
        img = np.asarray(img)
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
        return Image.fromarray(img)

def crop(img, top, left, height, width):
    if False:
        return 10
    'Crops the given PIL Image.\n\n    Args:\n        img (PIL.Image): Image to be cropped. (0,0) denotes the top left\n            corner of the image.\n        top (int): Vertical component of the top left corner of the crop box.\n        left (int): Horizontal component of the top left corner of the crop box.\n        height (int): Height of the crop box.\n        width (int): Width of the crop box.\n\n    Returns:\n        PIL.Image: Cropped image.\n\n    '
    return img.crop((left, top, left + width, top + height))

def center_crop(img, output_size):
    if False:
        print('Hello World!')
    "Crops the given PIL Image and resize it to desired size.\n\n    Args:\n        img (PIL.Image): Image to be cropped. (0,0) denotes the top left corner of the image.\n        output_size (sequence or int): (height, width) of the crop box. If int,\n            it is used for both directions\n        backend (str, optional): The image proccess backend type. Options are `pil`, `cv2`. Default: 'pil'.\n\n    Returns:\n        PIL.Image: Cropped image.\n\n    "
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    (image_width, image_height) = img.size
    (crop_height, crop_width) = output_size
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(img, crop_top, crop_left, crop_height, crop_width)

def hflip(img):
    if False:
        for i in range(10):
            print('nop')
    'Horizontally flips the given PIL Image.\n\n    Args:\n        img (PIL.Image): Image to be flipped.\n\n    Returns:\n        PIL.Image:  Horizontall flipped image.\n\n    '
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def vflip(img):
    if False:
        while True:
            i = 10
    'Vertically flips the given PIL Image.\n\n    Args:\n        img (PIL.Image): Image to be flipped.\n\n    Returns:\n        PIL.Image:  Vertically flipped image.\n\n    '
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def adjust_brightness(img, brightness_factor):
    if False:
        for i in range(10):
            print('nop')
    'Adjusts brightness of an Image.\n\n    Args:\n        img (PIL.Image): PIL Image to be adjusted.\n        brightness_factor (float):  How much to adjust the brightness. Can be\n            any non negative number. 0 gives a black image, 1 gives the\n            original image while 2 increases the brightness by a factor of 2.\n\n    Returns:\n        PIL.Image: Brightness adjusted image.\n\n    '
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img

def adjust_contrast(img, contrast_factor):
    if False:
        while True:
            i = 10
    'Adjusts contrast of an Image.\n\n    Args:\n        img (PIL.Image): PIL Image to be adjusted.\n        contrast_factor (float): How much to adjust the contrast. Can be any\n            non negative number. 0 gives a solid gray image, 1 gives the\n            original image while 2 increases the contrast by a factor of 2.\n\n    Returns:\n        PIL.Image: Contrast adjusted image.\n\n    '
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img

def adjust_saturation(img, saturation_factor):
    if False:
        return 10
    'Adjusts color saturation of an image.\n\n    Args:\n        img (PIL.Image): PIL Image to be adjusted.\n        saturation_factor (float):  How much to adjust the saturation. 0 will\n            give a black and white image, 1 will give the original image while\n            2 will enhance the saturation by a factor of 2.\n\n    Returns:\n        PIL.Image: Saturation adjusted image.\n\n    '
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img

def adjust_hue(img, hue_factor):
    if False:
        return 10
    'Adjusts hue of an image.\n\n    The image hue is adjusted by converting the image to HSV and\n    cyclically shifting the intensities in the hue channel (H).\n    The image is then converted back to original image mode.\n\n    `hue_factor` is the amount of shift in H channel and must be in the\n    interval `[-0.5, 0.5]`.\n\n    Args:\n        img (PIL.Image): PIL Image to be adjusted.\n        hue_factor (float):  How much to shift the hue channel. Should be in\n            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in\n            HSV space in positive and negative direction respectively.\n            0 means no shift. Therefore, both -0.5 and 0.5 will give an image\n            with complementary colors while 0 gives the original image.\n\n    Returns:\n        PIL.Image: Hue adjusted image.\n\n    '
    if not -0.5 <= hue_factor <= 0.5:
        raise ValueError(f'hue_factor:{hue_factor} is not in [-0.5, 0.5].')
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

def affine(img, matrix, interpolation='nearest', fill=0):
    if False:
        print('Hello World!')
    'Affine the image by matrix.\n\n    Args:\n        img (PIL.Image): Image to be affined.\n        matrix (float or int): Affine matrix.\n        interpolation (str, optional): Interpolation method. If omitted, or if the\n            image has only one channel, it is set to PIL.Image.NEAREST . when use pil backend,\n            support method are as following:\n            - "nearest": Image.NEAREST,\n            - "bilinear": Image.BILINEAR,\n            - "bicubic": Image.BICUBIC\n        fill (3-tuple or int): RGB pixel fill value for area outside the affined image.\n            If int, it is used for all channels respectively.\n\n    Returns:\n        PIL.Image: Affined image.\n\n    '
    if isinstance(fill, int):
        fill = tuple([fill] * 3)
    return img.transform(img.size, Image.AFFINE, matrix, _pil_interp_from_str[interpolation], fill)

def rotate(img, angle, interpolation='nearest', expand=False, center=None, fill=0):
    if False:
        for i in range(10):
            print('nop')
    'Rotates the image by angle.\n\n    Args:\n        img (PIL.Image): Image to be rotated.\n        angle (float or int): In degrees degrees counter clockwise order.\n        interpolation (str, optional): Interpolation method. If omitted, or if the\n            image has only one channel, it is set to PIL.Image.NEAREST . when use pil backend,\n            support method are as following:\n            - "nearest": Image.NEAREST,\n            - "bilinear": Image.BILINEAR,\n            - "bicubic": Image.BICUBIC\n        expand (bool, optional): Optional expansion flag.\n            If true, expands the output image to make it large enough to hold the entire rotated image.\n            If false or omitted, make the output image the same size as the input image.\n            Note that the expand flag assumes rotation around the center and no translation.\n        center (2-tuple, optional): Optional center of rotation.\n            Origin is the upper left corner.\n            Default is the center of the image.\n        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.\n            If int, it is used for all channels respectively.\n\n    Returns:\n        PIL.Image: Rotated image.\n\n    '
    if isinstance(fill, int):
        fill = tuple([fill] * 3)
    return img.rotate(angle, _pil_interp_from_str[interpolation], expand, center, fillcolor=fill)

def perspective(img, coeffs, interpolation='nearest', fill=0):
    if False:
        print('Hello World!')
    'Perspective the image.\n\n    Args:\n        img (PIL.Image): Image to be perspectived.\n        coeffs (list[float]): coefficients (a, b, c, d, e, f, g, h) of the perspective transforms.\n        interpolation (str, optional): Interpolation method. If omitted, or if the\n            image has only one channel, it is set to PIL.Image.NEAREST . when use pil backend,\n            support method are as following:\n            - "nearest": Image.NEAREST,\n            - "bilinear": Image.BILINEAR,\n            - "bicubic": Image.BICUBIC\n        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.\n            If int, it is used for all channels respectively.\n\n    Returns:\n        PIL.Image: Perspectived image.\n\n    '
    if isinstance(fill, int):
        fill = tuple([fill] * 3)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, _pil_interp_from_str[interpolation], fill)

def to_grayscale(img, num_output_channels=1):
    if False:
        for i in range(10):
            print('nop')
    "Converts image to grayscale version of image.\n\n    Args:\n        img (PIL.Image): Image to be converted to grayscale.\n        backend (str, optional): The image proccess backend type. Options are `pil`,\n                    `cv2`. Default: 'pil'.\n\n    Returns:\n        PIL.Image: Grayscale version of the image.\n            if num_output_channels = 1 : returned image is single channel\n\n            if num_output_channels = 3 : returned image is 3 channel with r = g = b\n\n    "
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

def erase(img, i, j, h, w, v, inplace=False):
    if False:
        return 10
    'Erase the pixels of selected area in input image with given value. PIL format is\n     not support inplace.\n\n    Args:\n         img (PIL.Image): input image, which shape is (C, H, W).\n         i (int): y coordinate of the top-left point of erased region.\n         j (int): x coordinate of the top-left point of erased region.\n         h (int): Height of the erased region.\n         w (int): Width of the erased region.\n         v (np.array): value used to replace the pixels in erased region.\n         inplace (bool, optional): Whether this transform is inplace. Default: False.\n\n     Returns:\n         PIL.Image: Erased image.\n\n    '
    np_img = np.array(img, dtype=np.uint8)
    np_img[i:i + h, j:j + w, ...] = v
    img = Image.fromarray(np_img, 'RGB')
    return img