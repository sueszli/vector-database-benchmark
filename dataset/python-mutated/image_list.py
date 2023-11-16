from __future__ import division
import torch

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes, pad_sizes):
        if False:
            i = 10
            return i + 15
        '\n        Arguments:\n            tensors (tensor)\n            image_sizes (list[tuple[int, int]])\n        '
        self.tensors = tensors
        self.image_sizes = image_sizes
        self.pad_sizes = pad_sizes

    def to(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes, self.pad_sizes)

def to_image_list(tensors, size_divisible=0, max_size=None):
    if False:
        i = 10
        return i + 15
    "\n    tensors can be an ImageList, a torch.Tensor or\n    an iterable of Tensors. It can't be a numpy array.\n    When tensors is an iterable of Tensors, it pads\n    the Tensors with zeros so that they have the same\n    shape\n    "
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]
    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        if max_size is None:
            max_size = tuple((max(s) for s in zip(*[img.shape for img in tensors])))
        if size_divisible > 0:
            import math
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)
        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for (img, pad_img) in zip(tensors, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        image_sizes = [im.shape[-2:] for im in tensors]
        pad_sizes = [batched_imgs.shape[-2:] for im in batched_imgs]
        return ImageList(batched_imgs, image_sizes, pad_sizes)
    else:
        raise TypeError('Unsupported type for to_image_list: {}'.format(type(tensors)))