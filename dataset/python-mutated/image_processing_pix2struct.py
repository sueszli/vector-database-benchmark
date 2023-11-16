"""Image processor class for Pix2Struct."""
import io
import math
from typing import Dict, Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import convert_to_rgb, normalize, to_channel_dimension_format, to_pil_image
from ...image_utils import ChannelDimension, ImageInput, get_image_size, infer_channel_dimension_format, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, is_torch_available, is_vision_available, logging
from ...utils.import_utils import requires_backends
if is_vision_available():
    import textwrap
    from PIL import Image, ImageDraw, ImageFont
if is_torch_available():
    import torch
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_11
else:
    is_torch_greater_or_equal_than_1_11 = False
logger = logging.get_logger(__name__)
DEFAULT_FONT_PATH = 'ybelkada/fonts'

def _check_torch_version():
    if False:
        for i in range(10):
            print('nop')
    if is_torch_available() and (not is_torch_greater_or_equal_than_1_11):
        raise ImportError(f'You are using torch=={torch.__version__}, but torch>=1.11.0 is required to use Pix2StructImageProcessor. Please upgrade torch.')

def torch_extract_patches(image_tensor, patch_height, patch_width):
    if False:
        print('Hello World!')
    '\n    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,\n    `patch_width`, `num_channels`x `patch_height` x `patch_width`)\n\n    Args:\n        image_tensor (torch.Tensor):\n            The image tensor to extract patches from.\n        patch_height (int):\n            The height of the patches to extract.\n        patch_width (int):\n            The width of the patches to extract.\n    '
    requires_backends(torch_extract_patches, ['torch'])
    _check_torch_version()
    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(image_tensor.size(2) // patch_height, image_tensor.size(3) // patch_width, image_tensor.size(1) * patch_height * patch_width)
    return patches.unsqueeze(0)

def render_text(text: str, text_size: int=36, text_color: str='black', background_color: str='white', left_padding: int=5, right_padding: int=5, top_padding: int=5, bottom_padding: int=5, font_bytes: Optional[bytes]=None, font_path: Optional[str]=None) -> Image.Image:
    if False:
        i = 10
        return i + 15
    '\n    Render text. This script is entirely adapted from the original script that can be found here:\n    https://github.com/google-research/pix2struct/blob/main/pix2struct/preprocessing/preprocessing_utils.py\n\n    Args:\n        text (`str`, *optional*, defaults to ):\n            Text to render.\n        text_size (`int`, *optional*, defaults to 36):\n            Size of the text.\n        text_color (`str`, *optional*, defaults to `"black"`):\n            Color of the text.\n        background_color (`str`, *optional*, defaults to `"white"`):\n            Color of the background.\n        left_padding (`int`, *optional*, defaults to 5):\n            Padding on the left.\n        right_padding (`int`, *optional*, defaults to 5):\n            Padding on the right.\n        top_padding (`int`, *optional*, defaults to 5):\n            Padding on the top.\n        bottom_padding (`int`, *optional*, defaults to 5):\n            Padding on the bottom.\n        font_bytes (`bytes`, *optional*):\n            Bytes of the font to use. If `None`, the default font will be used.\n        font_path (`str`, *optional*):\n            Path to the font to use. If `None`, the default font will be used.\n    '
    requires_backends(render_text, 'vision')
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=text)
    wrapped_text = '\n'.join(lines)
    if font_bytes is not None and font_path is None:
        font = io.BytesIO(font_bytes)
    elif font_path is not None:
        font = font_path
    else:
        font = hf_hub_download(DEFAULT_FONT_PATH, 'Arial.TTF')
    font = ImageFont.truetype(font, encoding='UTF-8', size=text_size)
    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1), background_color))
    (_, _, text_width, text_height) = temp_draw.textbbox((0, 0), wrapped_text, font)
    image_width = text_width + left_padding + right_padding
    image_height = text_height + top_padding + bottom_padding
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)
    draw.text(xy=(left_padding, top_padding), text=wrapped_text, fill=text_color, font=font)
    return image

def render_header(image: np.ndarray, header: str, input_data_format: Optional[Union[str, ChildProcessError]]=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Renders the input text as a header on the input image.\n\n    Args:\n        image (`np.ndarray`):\n            The image to render the header on.\n        header (`str`):\n            The header text.\n        data_format (`Union[ChannelDimension, str]`, *optional*):\n            The data format of the image. Can be either "ChannelDimension.channels_first" or\n            "ChannelDimension.channels_last".\n\n    Returns:\n        `np.ndarray`: The image with the header rendered.\n    '
    requires_backends(render_header, 'vision')
    image = to_pil_image(image, input_data_format=input_data_format)
    header_image = render_text(header, **kwargs)
    new_width = max(header_image.width, image.width)
    new_height = int(image.height * (new_width / image.width))
    new_header_height = int(header_image.height * (new_width / header_image.width))
    new_image = Image.new('RGB', (new_width, new_height + new_header_height), 'white')
    new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
    new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))
    new_image = to_numpy_array(new_image)
    if infer_channel_dimension_format(new_image) == ChannelDimension.LAST:
        new_image = to_channel_dimension_format(new_image, ChannelDimension.LAST)
    return new_image

class Pix2StructImageProcessor(BaseImageProcessor):
    """
    Constructs a Pix2Struct image processor.

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. According to Pix2Struct paper and code, the image is normalized with its own mean and standard
            deviation.
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image. According to Pix2Struct paper and code, the patch size is 16x16.
        max_patches (`int`, *optional*, defaults to 2048):
            The maximum number of patches to extract from the image as per the [Pix2Struct
            paper](https://arxiv.org/pdf/2210.03347.pdf).
        is_vqa (`bool`, *optional*, defaults to `False`):
            Whether or not the image processor is for the VQA task. If `True` and `header_text` is passed in, text is
            rendered onto the input images.
    """
    model_input_names = ['flattened_patches']

    def __init__(self, do_convert_rgb: bool=True, do_normalize: bool=True, patch_size: Dict[str, int]=None, max_patches: int=2048, is_vqa: bool=False, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.patch_size = patch_size if patch_size is not None else {'height': 16, 'width': 16}
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.max_patches = max_patches
        self.is_vqa = is_vqa

    def extract_flattened_patches(self, image: np.ndarray, max_patches: int, patch_size: dict, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract flattened patches from an image.\n\n        Args:\n            image (`np.ndarray`):\n                Image to extract flattened patches from.\n            max_patches (`int`):\n                Maximum number of patches to extract.\n            patch_size (`dict`):\n                Dictionary containing the patch height and width.\n\n        Returns:\n            result (`np.ndarray`):\n                A sequence of `max_patches` flattened patches.\n        '
        requires_backends(self.extract_flattened_patches, 'torch')
        _check_torch_version()
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)
        image = torch.from_numpy(image)
        (patch_height, patch_width) = (patch_size['height'], patch_size['width'])
        (image_height, image_width) = get_image_size(image, ChannelDimension.FIRST)
        scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(resized_height, resized_width), mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        patches = torch_extract_patches(image, patch_height, patch_width)
        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]
        patches = patches.reshape([rows * columns, depth])
        row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])
        row_ids += 1
        col_ids += 1
        row_ids = row_ids.to(torch.float32)
        col_ids = col_ids.to(torch.float32)
        result = torch.cat([row_ids, col_ids, patches], -1)
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - rows * columns]).float()
        result = to_numpy_array(result)
        return result

    def normalize(self, image: np.ndarray, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Normalize an image. image = (image - image_mean) / image_std.\n\n        The image std is to mimic the tensorflow implementation of the `per_image_standardization`:\n        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization\n\n        Args:\n            image (`np.ndarray`):\n                Image to normalize.\n            data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format for the output image. If unset, the channel dimension format of the input\n                image is used.\n            input_data_format (`str` or `ChannelDimension`, *optional*):\n                The channel dimension format of the input image. If not provided, it will be inferred.\n        '
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))
        return normalize(image, mean=mean, std=adjusted_stddev, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def preprocess(self, images: ImageInput, header_text: Optional[str]=None, do_convert_rgb: bool=None, do_normalize: Optional[bool]=None, max_patches: Optional[int]=None, patch_size: Optional[Dict[str, int]]=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> ImageInput:
        if False:
            i = 10
            return i + 15
        '\n        Preprocess an image or batch of images. The processor first computes the maximum possible number of\n        aspect-ratio preserving patches of size `patch_size` that can be extracted from the image. It then pads the\n        image with zeros to make the image respect the constraint of `max_patches`. Before extracting the patches the\n        images are standardized following the tensorflow implementation of `per_image_standardization`\n        (https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization).\n\n\n        Args:\n            images (`ImageInput`):\n                Image to preprocess. Expects a single or batch of images.\n            header_text (`Union[List[str], str]`, *optional*):\n                Text to render as a header. Only has an effect if `image_processor.is_vqa` is `True`.\n            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):\n                Whether to convert the image to RGB.\n            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):\n                Whether to normalize the image.\n            max_patches (`int`, *optional*, defaults to `self.max_patches`):\n                Maximum number of patches to extract.\n            patch_size (`dict`, *optional*, defaults to `self.patch_size`):\n                Dictionary containing the patch height and width.\n            return_tensors (`str` or `TensorType`, *optional*):\n                The type of tensors to return. Can be one of:\n                    - Unset: Return a list of `np.ndarray`.\n                    - `TensorType.TENSORFLOW` or `\'tf\'`: Return a batch of type `tf.Tensor`.\n                    - `TensorType.PYTORCH` or `\'pt\'`: Return a batch of type `torch.Tensor`.\n                    - `TensorType.NUMPY` or `\'np\'`: Return a batch of type `np.ndarray`.\n                    - `TensorType.JAX` or `\'jax\'`: Return a batch of type `jax.numpy.ndarray`.\n            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):\n                The channel dimension format for the output image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - Unset: Use the channel dimension format of the input image.\n            input_data_format (`ChannelDimension` or `str`, *optional*):\n                The channel dimension format for the input image. If unset, the channel dimension format is inferred\n                from the input image. Can be one of:\n                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.\n                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.\n                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.\n        '
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_size = patch_size if patch_size is not None else self.patch_size
        max_patches = max_patches if max_patches is not None else self.max_patches
        is_vqa = self.is_vqa
        if kwargs.get('data_format', None) is not None:
            raise ValueError('data_format is not an accepted input as the outputs are ')
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        images = [to_numpy_array(image) for image in images]
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if is_vqa:
            if header_text is None:
                raise ValueError('A header text must be provided for VQA models.')
            font_bytes = kwargs.pop('font_bytes', None)
            font_path = kwargs.pop('font_path', None)
            if isinstance(header_text, str):
                header_text = [header_text] * len(images)
            images = [render_header(image, header_text[i], font_bytes=font_bytes, font_path=font_path) for (i, image) in enumerate(images)]
        if do_normalize:
            images = [self.normalize(image=image, input_data_format=input_data_format) for image in images]
        images = [self.extract_flattened_patches(image=image, max_patches=max_patches, patch_size=patch_size, input_data_format=input_data_format) for image in images]
        attention_masks = [(image.sum(axis=-1) != 0).astype(np.float32) for image in images]
        encoded_outputs = BatchFeature(data={'flattened_patches': images, 'attention_mask': attention_masks}, tensor_type=return_tensors)
        return encoded_outputs