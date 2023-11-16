"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """
import sys
from typing import Tuple
import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers.image_utils import PILImageResampling
from utils import img_tensorize

class ResizeShortestEdge:

    def __init__(self, short_edge_length, max_size=sys.maxsize):
        if False:
            return 10
        '\n        Args:\n            short_edge_length (list[min, max])\n            max_size (int): maximum allowed longest edge length.\n        '
        self.interp_method = 'bilinear'
        self.max_size = max_size
        self.short_edge_length = short_edge_length

    def __call__(self, imgs):
        if False:
            i = 10
            return i + 15
        img_augs = []
        for img in imgs:
            (h, w) = img.shape[:2]
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            if size == 0:
                return img
            scale = size * 1.0 / min(h, w)
            if h < w:
                (newh, neww) = (size, scale * w)
            else:
                (newh, neww) = (scale * h, size)
            if max(newh, neww) > self.max_size:
                scale = self.max_size * 1.0 / max(newh, neww)
                newh = newh * scale
                neww = neww * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            if img.dtype == np.uint8:
                pil_image = Image.fromarray(img)
                pil_image = pil_image.resize((neww, newh), PILImageResampling.BILINEAR)
                img = np.asarray(pil_image)
            else:
                img = img.permute(2, 0, 1).unsqueeze(0)
                img = nn.functional.interpolate(img, (newh, neww), mode=self.interp_method, align_corners=False).squeeze(0)
            img_augs.append(img)
        return img_augs

class Preprocess:

    def __init__(self, cfg):
        if False:
            return 10
        self.aug = ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        self.size_divisibility = cfg.SIZE_DIVISIBILITY
        self.pad_value = cfg.PAD_VALUE
        self.max_image_size = cfg.INPUT.MAX_SIZE_TEST
        self.device = cfg.MODEL.DEVICE
        self.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(len(cfg.MODEL.PIXEL_STD), 1, 1)
        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(len(cfg.MODEL.PIXEL_STD), 1, 1)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

    def pad(self, images):
        if False:
            print('Hello World!')
        max_size = tuple((max(s) for s in zip(*[img.shape for img in images])))
        image_sizes = [im.shape[-2:] for im in images]
        images = [nn.functional.pad(im, [0, max_size[-1] - size[1], 0, max_size[-2] - size[0]], value=self.pad_value) for (size, im) in zip(image_sizes, images)]
        return (torch.stack(images), torch.tensor(image_sizes))

    def __call__(self, images, single_image=False):
        if False:
            for i in range(10):
                print('nop')
        with torch.no_grad():
            if not isinstance(images, list):
                images = [images]
            if single_image:
                assert len(images) == 1
            for i in range(len(images)):
                if isinstance(images[i], torch.Tensor):
                    images.insert(i, images.pop(i).to(self.device).float())
                elif not isinstance(images[i], torch.Tensor):
                    images.insert(i, torch.as_tensor(img_tensorize(images.pop(i), input_format=self.input_format)).to(self.device).float())
            raw_sizes = torch.tensor([im.shape[:2] for im in images])
            images = self.aug(images)
            images = [self.normalizer(x) for x in images]
            (images, sizes) = self.pad(images)
            if self.size_divisibility > 0:
                raise NotImplementedError()
            scales_yx = torch.true_divide(raw_sizes, sizes)
            if single_image:
                return (images[0], sizes[0], scales_yx[0])
            else:
                return (images, sizes, scales_yx)

def _scale_box(boxes, scale_yx):
    if False:
        i = 10
        return i + 15
    boxes[:, 0::2] *= scale_yx[:, 1]
    boxes[:, 1::2] *= scale_yx[:, 0]
    return boxes

def _clip_box(tensor, box_size: Tuple[int, int]):
    if False:
        for i in range(10):
            print('nop')
    assert torch.isfinite(tensor).all(), 'Box tensor contains infinite or NaN!'
    (h, w) = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)