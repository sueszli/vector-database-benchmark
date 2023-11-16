"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly available at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/datasets/pipelines
"""
import copy
import mmcv
import mmdet3d
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from numpy import random
from PIL import Image

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        if False:
            while True:
                i = 10
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if False:
            while True:
                i = 10
        'Pad images according to ``self.size``.'
        if self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        if False:
            i = 10
            return i + 15
        'Call function to pad images, masks, semantic segmentation maps.\n        Args:\n            results (dict): Result dict from loading pipeline.\n        Returns:\n            dict: Updated result dict.\n        '
        self._pad_img(results)
        return results

    def __repr__(self):
        if False:
            return 10
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        if False:
            i = 10
            return i + 15
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        if False:
            return 10
        "Call function to normalize images.\n        Args:\n            results (dict): Result dict from loading pipeline.\n        Returns:\n            dict: Normalized results, 'img_norm_cfg' key is added into\n                result dict.\n        "
        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        if False:
            while True:
                i = 10
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True, diff_aug=False):
        if False:
            for i in range(10):
                print('nop')
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.diff_aug = diff_aug

    def __call__(self, results):
        if False:
            for i in range(10):
                print('nop')
        'Call function to pad images, masks, semantic segmentation maps.\n        Args:\n            results (dict): Result dict from loading pipeline.\n        Returns:\n            dict: Updated result dict.\n        '
        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        results['intrin_ori'] = []
        results['extrin_ori'] = []
        results['ida_mat'] = []
        results['bda_mat'] = [np.eye(4)] * len(results['lidar2img'])
        if not self.diff_aug:
            (resize, resize_dims, crop, flip, rotate) = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            if self.diff_aug:
                (resize, resize_dims, crop, flip, rotate) = self._sample_augmentation()
            (img, ida_mat) = self._img_transform(img, resize=resize, resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate)
            new_imgs.append(np.array(img).astype(np.float32))
            results['intrin_ori'] += [results['intrinsics'][i][:3, :3].copy()]
            results['extrin_ori'] += [results['extrinsics'][i].T.copy()]
            results['ida_mat'] += [ida_mat.cpu().numpy()]
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i].T for i in range(len(results['extrinsics']))]
        return results

    def _get_rot(self, h):
        if False:
            i = 10
            return i + 15
        return torch.Tensor([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        if False:
            print('Hello World!')
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return (img, ida_mat)

    def _sample_augmentation(self):
        if False:
            while True:
                i = 10
        (H, W) = (self.data_aug_conf['H'], self.data_aug_conf['W'])
        (fH, fW) = self.data_aug_conf['final_dim']
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            (newW, newH) = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            (newW, newH) = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return (resize, resize_dims, crop, flip, rotate)