import collections
import math
import numbers
import random
import cv2
import numpy as np
import torch
_cv2_pad_to_str = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE, 'reflect': cv2.BORDER_REFLECT_101, 'symmetric': cv2.BORDER_REFLECT}
_cv2_interpolation_to_str = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA, 'bicubic': cv2.INTER_CUBIC, 'lanczos': cv2.INTER_LANCZOS4}
_cv2_interpolation_from_str = {v: k for (k, v) in _cv2_interpolation_to_str.items()}

def _is_tensor_image(img):
    if False:
        for i in range(10):
            print('nop')
    return torch.is_tensor(img) and img.ndimension() == 3

def _is_numpy_image(img):
    if False:
        print('Hello World!')
    return isinstance(img, np.ndarray) and img.ndim in {2, 3}

def to_tensor(pic):
    if False:
        for i in range(10):
            print('nop')
    'Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.\n    See ``ToTensor`` for more details.\n    Args:\n        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.\n    Returns:\n        Tensor: Converted image.\n    '
    if not _is_numpy_image(pic):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    if isinstance(img, torch.ByteTensor) or img.dtype == torch.uint8:
        return img.float().div(255)
    else:
        return img

def normalize(tensor, mean, std):
    if False:
        for i in range(10):
            print('nop')
    'Normalize a tensor image with mean and standard deviation.\n    .. note::\n        This transform acts in-place, i.e., it mutates the input tensor.\n    See :class:`~torchvision.transforms.Normalize` for more details.\n    Args:\n        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.\n        mean (sequence): Sequence of means for each channel.\n        std (sequence): Sequence of standard deviations for each channely.\n    Returns:\n        Tensor: Normalized Tensor image.\n    '
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    for (t, m, s) in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def resize(img, size, interpolation=cv2.INTER_LINEAR):
    if False:
        i = 10
        return i + 15
    'Resize the input numpy ndarray to the given size.\n    Args:\n        img (numpy ndarray): Image to be resized.\n        size (sequence or int): Desired output size. If size is a sequence like\n            (h, w), the output size will be matched to this. If size is an int,\n            the smaller edge of the image will be matched to this number maintaing\n            the aspect ratio. i.e, if height > width, then image will be rescaled to\n            :math:`\\left(\\text{size} \\times \\frac{\\text{height}}{\\text{width}}, \\text{size}\\right)`\n        interpolation (int, optional): Desired interpolation. Default is\n            ``cv2.INTER_LINEAR``\n    Returns:\n        PIL Image: Resized image.\n    '
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    (h, w) = (img.shape[0], img.shape[1])
    if isinstance(size, int):
        if w <= h and w == size or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        (ow, oh) = (size[1], size[0])
    output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
    if img.shape[2] == 1:
        return output[:, :, np.newaxis]
    else:
        return output

def pad(img, padding, fill=0, padding_mode='constant'):
    if False:
        print('Hello World!')
    'Pad the given numpy ndarray on all sides with specified padding mode and fill value.\n    Args:\n        img (numpy ndarray): image to be padded.\n        padding (int or tuple): Padding on each border. If a single int is provided this\n            is used to pad all borders. If tuple of length 2 is provided this is the padding\n            on left/right and top/bottom respectively. If a tuple of length 4 is provided\n            this is the padding for the left, top, right and bottom borders\n            respectively.\n        fill: Pixel fill value for constant fill. Default is 0. If a tuple of\n            length 3, it is used to fill R, G, B channels respectively.\n            This value is only used when the padding_mode is constant\n        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.\n            - constant: pads with a constant value, this value is specified with fill\n            - edge: pads with the last value on the edge of the image\n            - reflect: pads with reflection of image (without repeating the last value on the edge)\n                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode\n                       will result in [3, 2, 1, 2, 3, 4, 3, 2]\n            - symmetric: pads with reflection of image (repeating the last value on the edge)\n                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode\n                         will result in [2, 1, 1, 2, 3, 4, 4, 3]\n    Returns:\n        Numpy image: padded image.\n    '
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError('Padding must be an int or a 2, or 4 element tuple, not a ' + '{} element tuple'.format(len(padding)))
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], 'Padding mode should be either constant, edge, reflect or symmetric'
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    if img.shape[2] == 1:
        return cv2.copyMakeBorder(img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, borderType=_cv2_pad_to_str[padding_mode], value=fill)[:, :, np.newaxis]
    else:
        return cv2.copyMakeBorder(img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, borderType=_cv2_pad_to_str[padding_mode], value=fill)

def crop(img, i, j, h, w):
    if False:
        i = 10
        return i + 15
    'Crop the given PIL Image.\n    Args:\n        img (numpy ndarray): Image to be cropped.\n        i: Upper pixel coordinate.\n        j: Left pixel coordinate.\n        h: Height of the cropped image.\n        w: Width of the cropped image.\n    Returns:\n        numpy ndarray: Cropped image.\n    '
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    return img[i:i + h, j:j + w, :]

def center_crop(img, output_size):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    (h, w) = img.shape[0:2]
    (th, tw) = output_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(img, i, j, th, tw)

def resized_crop(img, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
    if False:
        for i in range(10):
            print('nop')
    'Crop the given numpy ndarray and resize it to desired size.\n    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.\n    Args:\n        img (numpy ndarray): Image to be cropped.\n        i: Upper pixel coordinate.\n        j: Left pixel coordinate.\n        h: Height of the cropped image.\n        w: Width of the cropped image.\n        size (sequence or int): Desired output size. Same semantics as ``scale``.\n        interpolation (int, optional): Desired interpolation. Default is\n            ``cv2.INTER_CUBIC``.\n    Returns:\n        PIL Image: Cropped image.\n    '
    assert _is_numpy_image(img), 'img should be numpy image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation=interpolation)
    return img

def hflip(img):
    if False:
        for i in range(10):
            print('nop')
    'Horizontally flip the given numpy ndarray.\n    Args:\n        img (numpy ndarray): image to be flipped.\n    Returns:\n        numpy ndarray:  Horizontally flipped image.\n    '
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if img.shape[2] == 1:
        return cv2.flip(img, 1)[:, :, np.newaxis]
    else:
        return cv2.flip(img, 1)

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.\n        Returns:\n            Tensor: Converted image.\n        '
        return to_tensor(pic)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__ + '()'

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        if False:
            print('Hello World!')
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if False:
            return 10
        '\n        Args:\n            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.\n        Returns:\n            Tensor: Normalized Tensor image.\n        '
        return normalize(tensor, self.mean, self.std)

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Resize(object):
    """Resize the input numpy ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``, bicubic interpolation
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        if False:
            while True:
                i = 10
        if isinstance(size, int):
            self.size = size
        elif isinstance(size, collections.abc.Iterable) and len(size) == 2:
            if type(size) == list:
                size = tuple(size)
            self.size = size
        else:
            raise ValueError('Unknown inputs for size: {}'.format(size))
        self.interpolation = interpolation

    def __call__(self, img):
        if False:
            print('Hello World!')
        '\n        Args:\n            img (numpy ndarray): Image to be scaled.\n        Returns:\n            numpy ndarray: Rescaled image.\n        '
        return resize(img, self.size, self.interpolation)

    def __repr__(self):
        if False:
            while True:
                i = 10
        interpolate_str = _cv2_interpolation_from_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class CenterCrop(object):
    """Crops the given numpy ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if False:
            print('Hello World!')
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        if False:
            print('Hello World!')
        '\n        Args:\n            img (numpy ndarray): Image to be cropped.\n        Returns:\n            numpy ndarray: Cropped image.\n        '
        return center_crop(img, self.size)

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomCrop(object):
    """Crop the given numpy ndarray at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        if False:
            while True:
                i = 10
        'Get parameters for ``crop`` for a random crop.\n        Args:\n            img (numpy ndarray): Image to be cropped.\n            output_size (tuple): Expected output size of the crop.\n        Returns:\n            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.\n        '
        (h, w) = img.shape[0:2]
        (th, tw) = output_size
        if w == tw and h == th:
            return (0, 0, h, w)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return (i, j, th, tw)

    def __call__(self, img):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            img (numpy ndarray): Image to be cropped.\n        Returns:\n            numpy ndarray: Cropped image.\n        '
        if self.padding is not None:
            img = pad(img, self.padding, self.fill, self.padding_mode)
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = pad(img, (self.size[1] - img.shape[1], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = pad(img, (0, self.size[0] - img.shape[0]), self.fill, self.padding_mode)
        (i, j, h, w) = self.get_params(img, self.size)
        return crop(img, i, j, h, w)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class RandomResizedCrop(object):
    """Crop the given numpy ndarray to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: cv2.INTER_CUBIC
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=cv2.INTER_LINEAR):
        if False:
            i = 10
            return i + 15
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        if False:
            i = 10
            return i + 15
        'Get parameters for ``crop`` for a random sized crop.\n        Args:\n            img (numpy ndarray): Image to be cropped.\n            scale (tuple): range of size of the origin size cropped\n            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped\n        Returns:\n            tuple: params (i, j, h, w) to be passed to ``crop`` for a random\n                sized crop.\n        '
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                (w, h) = (h, w)
            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return (i, j, h, w)
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return (i, j, w, w)

    def __call__(self, img):
        if False:
            return 10
        '\n        Args:\n            img (numpy ndarray): Image to be cropped and resized.\n        Returns:\n            numpy ndarray: Randomly cropped and resized image.\n        '
        (i, j, h, w) = self.get_params(img, self.scale, self.ratio)
        return resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        if False:
            print('Hello World!')
        interpolate_str = _cv2_interpolation_from_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple((round(s, 4) for s in self.scale)))
        format_string += ', ratio={0}'.format(tuple((round(r, 4) for r in self.ratio)))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        if False:
            i = 10
            return i + 15
        self.p = p

    def __call__(self, img):
        if False:
            return 10
        'random\n        Args:\n            img (numpy ndarray): Image to be flipped.\n        Returns:\n            numpy ndarray: Randomly flipped image.\n        '
        if random.random() < self.p:
            return hflip(img)
        return img

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__ + '(p={})'.format(self.p)