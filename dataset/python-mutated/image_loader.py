from os import PathLike
from typing import Union, Sequence, Tuple, List, cast
import torch
import torchvision
from torch import FloatTensor, IntTensor
from allennlp.common.file_utils import cached_path
from allennlp.common.registrable import Registrable
OnePath = Union[str, PathLike]
ManyPaths = Sequence[OnePath]
ImagesWithSize = Tuple[FloatTensor, IntTensor]

class ImageLoader(Registrable):
    """
    An `ImageLoader` is a callable that takes as input one or more filenames, and outputs two
    tensors: one representing the images themselves, and one that just holds the sizes
    of each image.

    The first tensor is the images and is of shape `(batch_size, color_channels, height, width)`.
    The second tensor is the sizes and is of shape `(batch_size, 2)`, where
    the last dimension contains the height and width, respectively.

    If only a single image is passed (as a `Path` or `str`, instead of a list) then
    the batch dimension will be removed.

    Subclasses only need to implement the `load()` method, which should load a single image
    from a path.

    # Parameters

    size_divisibility : `int`, optional (default = `0`)
        If set to a positive number, padding will be added so that the height
        and width dimensions are divisible by `size_divisibility`.
        Certain models may require this.

    pad_value : `float`, optional (default = `0.0`)
        The value to use for padding.

    device : `Union[str, torch.device]`, optional (default = `"cpu"`)
        A torch device identifier to put the image and size tensors on.
    """
    default_implementation = 'torch'

    def __init__(self, *, size_divisibility: int=0, pad_value: float=0.0, device: Union[str, torch.device]='cpu') -> None:
        if False:
            print('Hello World!')
        self.size_divisibility = size_divisibility
        self.pad_value = pad_value
        self.device = device

    def __call__(self, filename_or_filenames: Union[OnePath, ManyPaths]) -> ImagesWithSize:
        if False:
            print('Hello World!')
        if not isinstance(filename_or_filenames, (list, tuple)):
            (image, size) = self([filename_or_filenames])
            return (cast(FloatTensor, image.squeeze(0)), cast(IntTensor, size.squeeze(0)))
        images: List[FloatTensor] = []
        sizes: List[IntTensor] = []
        for filename in filename_or_filenames:
            image = self.load(cached_path(filename)).to(self.device)
            size = cast(IntTensor, torch.tensor([image.shape[-2], image.shape[-1]], dtype=torch.int32, device=self.device))
            images.append(image)
            sizes.append(size)
        return self._pack_image_list(images, sizes)

    def load(self, filename: OnePath) -> FloatTensor:
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def _pack_image_list(self, images: List[FloatTensor], sizes: List[IntTensor]) -> ImagesWithSize:
        if False:
            i = 10
            return i + 15
        '\n        A helper method that subclasses can use to turn a list of individual images into a padded\n        batch.\n        '
        size_tensor = torch.stack(sizes)
        max_size = size_tensor.max(0).values
        if self.size_divisibility > 1:
            max_size = (max_size + self.size_divisibility - 1) // self.size_divisibility * self.size_divisibility
        batched_shape = [len(images)] + list(images[0].shape[:-2]) + list(max_size)
        batched_images = images[0].new_full(batched_shape, self.pad_value)
        for (image, batch_slice, size) in zip(images, batched_images, size_tensor):
            batch_slice[..., :image.shape[-2], :image.shape[-1]].copy_(image)
        return (cast(FloatTensor, batched_images), cast(IntTensor, size_tensor))

@ImageLoader.register('torch')
class TorchImageLoader(ImageLoader):
    """
    This is just a wrapper around the default image loader from [torchvision]
    (https://pytorch.org/docs/stable/torchvision/io.html#image).

    # Parameters

    image_backend : `Optional[str]`, optional (default = `None`)
        Set the image backend. Can be one of `"PIL"` or `"accimage"`.
    resize : `bool`, optional (default = `True`)
        If `True` (the default), images will be resized when necessary according
        to the values of `min_size` and `max_size`.
    normalize: `bool`, optional (default = `True`)
        If `True` (the default), images will be normalized according to the values
        of `pixel_mean` and `pixel_std`.
    min_size : `int`, optional (default = `800`)
        If `resize` is `True`, images smaller than this will be resized up to `min_size`.
    max_size : `int`, optional (default = `1333`)
        If `resize` is `True`, images larger than this will be resized down to `max_size`.
    pixel_mean : `Tuple[float, float, float]`, optional (default = `(0.485, 0.456, 0.406)`)
        Mean values for image normalization. The defaults are reasonable for most models
        from `torchvision`.
    pixel_std : `Tuple[float, float, float]`, optional (default = `(0.229, 0.224, 0.225)`)
        Standard deviation for image normalization. The defaults are reasonable for most
        models from `torchvision`.
    size_divisibility : `int`, optional (default = `32`)
        Same parameter as with the `ImageLoader` base class, but the default here is
        different.
    """

    def __init__(self, *, image_backend: str=None, resize: bool=True, normalize: bool=True, min_size: int=800, max_size: int=1333, pixel_mean: Tuple[float, float, float]=(0.485, 0.456, 0.406), pixel_std: Tuple[float, float, float]=(0.229, 0.224, 0.225), size_divisibility: int=32, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(size_divisibility=size_divisibility, **kwargs)
        if image_backend is not None:
            torchvision.set_image_backend(image_backend)
        self.resize = resize
        self.normalize = normalize
        self.min_size = min_size
        self.max_size = max_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def load(self, filename: OnePath) -> FloatTensor:
        if False:
            for i in range(10):
                print('nop')
        image = torchvision.io.read_image(filename).float().to(self.device) / 256
        if self.normalize:
            mean = torch.as_tensor(self.pixel_mean, dtype=image.dtype, device=self.device).view(-1, 1, 1)
            std = torch.as_tensor(self.pixel_std, dtype=image.dtype, device=self.device).view(-1, 1, 1)
            image = (image - mean) / std
        if self.resize:
            min_size = min(image.shape[-2:])
            max_size = max(image.shape[-2:])
            scale_factor = self.min_size / min_size
            if max_size * scale_factor > self.max_size:
                scale_factor = self.max_size / max_size
            image = torch.nn.functional.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)[0]
        return image