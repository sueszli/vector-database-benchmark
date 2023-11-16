import math
import numbers
import random
import torch.nn as nn
from . import functional_video as F
__all__ = ['ResizeVideo', 'RandomCropVideo', 'RandomResizedCropVideo', 'CenterCropVideo', 'NormalizeVideo', 'ToTensorVideo', 'RandomHorizontalFlipVideo']

class ResizeVideo(object):

    def __init__(self, size, keep_ratio=True, interpolation_mode='bilinear'):
        if False:
            i = 10
            return i + 15
        if isinstance(size, tuple):
            assert len(size) == 2, 'size should be tuple (height, width)'
        self.size = size
        self.keep_ratio = keep_ratio
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        if False:
            for i in range(10):
                print('nop')
        (size, scale) = (None, None)
        if isinstance(self.size, numbers.Number):
            if self.keep_ratio:
                scale = self.size / min(clip.shape[-2:])
            else:
                size = (int(self.size), int(self.size))
        elif self.keep_ratio:
            scale = min(self.size[0] / clip.shape[-2], self.size[1] / clip.shape[-1])
        else:
            size = self.size
        return nn.functional.interpolate(clip, size=size, scale_factor=scale, mode=self.interpolation_mode, align_corners=False)

class RandomCropVideo(object):

    def __init__(self, size):
        if False:
            while True:
                i = 10
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W).\n        Returns:\n            torch.tensor: randomly cropped/resized video clip.\n                size is (C, T, OH, OW)\n        '
        (i, j, h, w) = self.get_params(clip, self.size)
        return F.crop(clip, i, j, h, w)

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    @staticmethod
    def get_params(clip, output_size):
        if False:
            print('Hello World!')
        'Get parameters for ``crop`` for a random crop.\n\n        Args:\n            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W).\n            output_size (tuple): Expected output size of the crop.\n\n        Returns:\n            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.\n        '
        (w, h) = (clip.shape[3], clip.shape[2])
        (th, tw) = output_size
        if w == tw and h == th:
            return (0, 0, h, w)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return (i, j, th, tw)

class RandomResizedCropVideo(object):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation_mode='bilinear'):
        if False:
            while True:
                i = 10
        if isinstance(size, tuple):
            assert len(size) == 2, 'size should be tuple (height, width)'
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip):
        if False:
            while True:
                i = 10
        '\n        Args:\n            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W).\n        Returns:\n            torch.tensor: randomly cropped/resized video clip.\n                size is (C, T, H, W)\n        '
        (i, j, h, w) = self.get_params(clip, self.scale, self.ratio)
        return F.resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode)

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__class__.__name__ + '(size={0}, interpolation_mode={1}, scale={2}, ratio={3})'.format(self.size, self.interpolation_mode, self.scale, self.ratio)

    @staticmethod
    def get_params(clip, scale, ratio):
        if False:
            while True:
                i = 10
        'Get parameters for ``crop`` for a random sized crop.\n\n        Args:\n            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)\n            scale (tuple): range of size of the origin size cropped\n            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped\n\n        Returns:\n            tuple: params (i, j, h, w) to be passed to ``crop`` for a random\n                sized crop.\n        '
        (_w, _h) = (clip.shape[3], clip.shape[2])
        area = _w * _h
        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if w <= _w and h <= _h:
                i = random.randint(0, _h - h)
                j = random.randint(0, _w - w)
                return (i, j, h, w)
        in_ratio = _w / _h
        if in_ratio < min(ratio):
            w = _w
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = _h
            w = int(round(h * max(ratio)))
        else:
            w = _w
            h = _h
        i = (_h - h) // 2
        j = (_w - w) // 2
        return (i, j, h, w)

class CenterCropVideo(object):

    def __init__(self, size):
        if False:
            print('Hello World!')
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        if False:
            print('Hello World!')
        '\n        Args:\n            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)\n        Returns:\n            torch.tensor: central cropping of video clip. Size is\n            (C, T, size, size)\n        '
        return F.center_crop(clip, self.size)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        if False:
            print('Hello World!')
        '\n        Args:\n            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)\n        '
        return F.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        if False:
            return 10
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(self.mean, self.std, self.inplace)

class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def __call__(self, clip):
        if False:
            while True:
                i = 10
        '\n        Args:\n            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)\n        Return:\n            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)\n        '
        return F.to_tensor(clip)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

class RandomHorizontalFlipVideo(object):
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        if False:
            while True:
                i = 10
        self.p = p

    def __call__(self, clip):
        if False:
            while True:
                i = 10
        '\n        Args:\n            clip (torch.tensor): Size is (C, T, H, W)\n        Return:\n            clip (torch.tensor): Size is (C, T, H, W)\n        '
        if random.random() < self.p:
            clip = F.hflip(clip)
        return clip

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__.__name__ + '(p={0})'.format(self.p)