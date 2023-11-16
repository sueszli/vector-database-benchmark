import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch

class GroupScale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        if False:
            print('Hello World!')
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        if False:
            while True:
                i = 10
        return [self.worker(img) for img in img_group]

class GroupRandomCrop(object):

    def __init__(self, size):
        if False:
            return 10
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        if False:
            print('Hello World!')
        (w, h) = img_group[0].size
        (th, tw) = self.size
        out_images = list()
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return out_images

class GroupCenterCrop(object):

    def __init__(self, size):
        if False:
            return 10
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        if False:
            i = 10
            return i + 15
        cropped_imgs = [self.worker(img) for img in img_group]
        return cropped_imgs

class GroupRandomHorizontalFlip(object):

    def __call__(self, img_group):
        if False:
            return 10
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group

class GroupNormalize(object):

    def __init__(self, modality, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        if False:
            i = 10
            return i + 15
        self.modality = modality
        self.means = means
        self.stds = stds
        self.tensor_worker = torchvision.transforms.ToTensor()
        self.norm_worker = torchvision.transforms.Normalize(mean=means, std=stds)

    def __call__(self, img_group):
        if False:
            for i in range(10):
                print('nop')
        if self.modality == 'RGB':
            img_tensors = [self.tensor_worker(img) for img in img_group]
            img_tensors = [self.norm_worker(img) for img in img_tensors]
        else:
            img_arrays = [np.asarray(img).transpose([2, 0, 1]) for img in img_group]
            img_tensors = [torch.from_numpy(img / 255.0 * 2 - 1) for img in img_arrays]
        return img_tensors

class Stack(object):

    def __call__(self, img_tensors):
        if False:
            for i in range(10):
                print('nop')
        return torch.stack(img_tensors, dim=0).permute(1, 0, 2, 3).float()