"""
Adversarial perturbations designed to work for images.
"""
from typing import Optional, Tuple
import numpy as np

def add_single_bd(x: np.ndarray, distance: int=2, pixel_value: int=1) -> np.ndarray:
    if False:
        return 10
    '\n    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for a single image\n    or a batch of images.\n\n    :param x: A single image or batch of images of shape NWHC, NHW, or HC. Pixels will be added to all channels.\n    :param distance: Distance from bottom-right walls.\n    :param pixel_value: Value used to replace the entries of the image matrix.\n    :return: Backdoored image.\n    '
    x = np.copy(x)
    shape = x.shape
    if len(shape) == 4:
        (height, width) = x.shape[1:3]
        x[:, height - distance, width - distance, :] = pixel_value
    elif len(shape) == 3:
        (height, width) = x.shape[1:]
        x[:, height - distance, width - distance] = pixel_value
    elif len(shape) == 2:
        (height, width) = x.shape
        x[height - distance, width - distance] = pixel_value
    else:
        raise ValueError(f'Invalid array shape: {shape}')
    return x

def add_pattern_bd(x: np.ndarray, distance: int=2, pixel_value: int=1, channels_first: bool=False) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Augments a matrix by setting a checkerboard-like pattern of values some `distance` away from the bottom-right\n    edge to 1. Works for single images or a batch of images.\n\n    :param x: A single image or batch of images of shape NWHC, NHW, or HC. Pixels will be added to all channels.\n    :param distance: Distance from bottom-right walls.\n    :param pixel_value: Value used to replace the entries of the image matrix.\n    :param channels_first: If the data is provided in channels first format we transpose to NWHC or HC depending on\n                           input shape\n    :return: Backdoored image.\n    '
    x = np.copy(x)
    original_dtype = x.dtype
    shape = x.shape
    if channels_first:
        if len(shape) == 4:
            x = np.transpose(x, (0, 2, 3, 1))
        if len(shape) == 2:
            x = np.transpose(x)
    if len(shape) == 4:
        (height, width) = x.shape[1:3]
        x[:, height - distance, width - distance, :] = pixel_value
        x[:, height - distance - 1, width - distance - 1, :] = pixel_value
        x[:, height - distance, width - distance - 2, :] = pixel_value
        x[:, height - distance - 2, width - distance, :] = pixel_value
    elif len(shape) == 3:
        (height, width) = x.shape[1:]
        x[:, height - distance, width - distance] = pixel_value
        x[:, height - distance - 1, width - distance - 1] = pixel_value
        x[:, height - distance, width - distance - 2] = pixel_value
        x[:, height - distance - 2, width - distance] = pixel_value
    elif len(shape) == 2:
        (height, width) = x.shape
        x[height - distance, width - distance] = pixel_value
        x[height - distance - 1, width - distance - 1] = pixel_value
        x[height - distance, width - distance - 2] = pixel_value
        x[height - distance - 2, width - distance] = pixel_value
    else:
        raise ValueError(f'Invalid array shape: {shape}')
    if channels_first:
        if len(shape) == 4:
            x = np.transpose(x, (0, 3, 1, 2))
        if len(shape) == 2:
            x = np.transpose(x)
    return x.astype(original_dtype)

def insert_image(x: np.ndarray, backdoor_path: str='../utils/data/backdoors/alert.png', channels_first: bool=False, random: bool=True, x_shift: int=0, y_shift: int=0, size: Optional[Tuple[int, int]]=None, mode: str='L', blend=0.8) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Augments a matrix by setting a checkerboard-like pattern of values some `distance` away from the bottom-right\n    edge to 1. Works for single images or a batch of images.\n\n    :param x: A single image or batch of images of shape NHWC, NCHW, or HWC. Input is in range [0,1].\n    :param backdoor_path: The path to the image to insert as a trigger.\n    :param channels_first: Whether the channels axis is in the first or last dimension\n    :param random: Whether or not the image should be randomly placed somewhere on the image.\n    :param x_shift: Number of pixels from the left to shift the trigger (when not using random placement).\n    :param y_shift: Number of pixels from the right to shift the trigger (when not using random placement).\n    :param size: The size the trigger image should be (height, width). Default `None` if no resizing necessary.\n    :param mode: The mode the image should be read in. See PIL documentation\n                 (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes).\n    :param blend: The blending factor\n    :return: Backdoored image.\n    '
    from PIL import Image
    n_dim = len(x.shape)
    if n_dim == 4:
        return np.array([insert_image(single_img, backdoor_path, channels_first, random, x_shift, y_shift, size, mode, blend) for single_img in x])
    if n_dim != 3:
        raise ValueError(f'Invalid array shape {x.shape}')
    original_dtype = x.dtype
    data = np.copy(x)
    if channels_first:
        data = np.transpose(data, (1, 2, 0))
    (height, width, num_channels) = data.shape
    no_color = num_channels == 1
    orig_img = Image.new('RGBA', (width, height), 0)
    backdoored_img = Image.new('RGBA', (width, height), 0)
    if no_color:
        backdoored_input = Image.fromarray((data * 255).astype(np.uint8).squeeze(axis=2), mode=mode)
    else:
        backdoored_input = Image.fromarray((data * 255).astype(np.uint8), mode=mode)
    orig_img.paste(backdoored_input)
    trigger = Image.open(backdoor_path).convert('RGBA')
    if size is not None:
        trigger = trigger.resize((size[1], size[0]))
    (backdoor_width, backdoor_height) = trigger.size
    if backdoor_width > width or backdoor_height > height:
        raise ValueError('Backdoor does not fit inside original image')
    if random:
        x_shift = np.random.randint(width - backdoor_width + 1)
        y_shift = np.random.randint(height - backdoor_height + 1)
    backdoored_img.paste(trigger, (x_shift, y_shift), mask=trigger)
    composite = Image.alpha_composite(orig_img, backdoored_img)
    backdoored_img = Image.blend(orig_img, composite, blend)
    backdoored_img = backdoored_img.convert(mode)
    res = np.asarray(backdoored_img) / 255.0
    if no_color:
        res = np.expand_dims(res, 2)
    if channels_first:
        res = np.transpose(res, (2, 0, 1))
    return res.astype(original_dtype)