try:
    import numpy as np
except ImportError:
    raise RuntimeError("Could not import numpy. DALI's automatic augmentation examples depend on numpy. Please install numpy to use the examples.")
from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali.auto_aug.core import augmentation
"\nThis module contains a standard suite of augmentations used by AutoAugment policy for ImageNet,\nRandAugment and TrivialAugmentWide. The augmentations are implemented in terms of DALI operators.\n\nThe `@augmentation` decorator handles computation of the decorated transformations's parameter.\nWhen called, the decorated augmentation expects:\n* a single positional argument: batch of samples\n* `magnitude_bin` and `num_magnitude_bins` instead of the parameter.\n  The parameter is computed as if by calling\n  `mag_to_param(magnitudes[magnitude_bin] * ((-1) ** random_sign))`, where\n  `magnitudes=linspace(mag_range[0], mag_range[1], num_magnitude_bins)`.\n\nThe augmentations in this module are defined with example setups passed\nto `@augmentation`. The parameters can be easily adjusted. For instance, to increase\nthe magnitudes range of `shear_x` from 0.3 to 0.5, you can create\n`my_shear_x = shear_x.augmentation(mag_range=(0, 0.5))`.\n"

def warp_x_param(magnitude):
    if False:
        for i in range(10):
            print('nop')
    return [magnitude, 0]

def warp_y_param(magnitude):
    if False:
        i = 10
        return i + 15
    return [0, magnitude]

@augmentation(mag_range=(0, 0.3), randomly_negate=True, mag_to_param=warp_x_param)
def shear_x(data, shear, fill_value=128, interp_type=None):
    if False:
        i = 10
        return i + 15
    mt = fn.transforms.shear(shear=shear)
    return fn.warp_affine(data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False)

@augmentation(mag_range=(0, 0.3), randomly_negate=True, mag_to_param=warp_y_param)
def shear_y(data, shear, fill_value=128, interp_type=None):
    if False:
        return 10
    mt = fn.transforms.shear(shear=shear)
    return fn.warp_affine(data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False)

@augmentation(mag_range=(0.0, 1.0), randomly_negate=True, mag_to_param=warp_x_param)
def translate_x(data, rel_offset, shape, fill_value=128, interp_type=None):
    if False:
        print('Hello World!')
    offset = rel_offset * shape[1]
    mt = fn.transforms.translation(offset=offset)
    return fn.warp_affine(data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False)

@augmentation(mag_range=(0, 250), randomly_negate=True, mag_to_param=warp_x_param, name='translate_x')
def translate_x_no_shape(data, offset, fill_value=128, interp_type=None):
    if False:
        return 10
    mt = fn.transforms.translation(offset=offset)
    return fn.warp_affine(data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False)

@augmentation(mag_range=(0.0, 1.0), randomly_negate=True, mag_to_param=warp_y_param)
def translate_y(data, rel_offset, shape, fill_value=128, interp_type=None):
    if False:
        print('Hello World!')
    offset = rel_offset * shape[0]
    mt = fn.transforms.translation(offset=offset)
    return fn.warp_affine(data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False)

@augmentation(mag_range=(0, 250), randomly_negate=True, mag_to_param=warp_y_param, name='translate_y')
def translate_y_no_shape(data, offset, fill_value=128, interp_type=None):
    if False:
        return 10
    mt = fn.transforms.translation(offset=offset)
    return fn.warp_affine(data, matrix=mt, fill_value=fill_value, interp_type=interp_type, inverse_map=False)

@augmentation(mag_range=(0, 30), randomly_negate=True)
def rotate(data, angle, fill_value=128, interp_type=None, rotate_keep_size=True):
    if False:
        while True:
            i = 10
    return fn.rotate(data, angle=angle, fill_value=fill_value, interp_type=interp_type, keep_size=rotate_keep_size)

def shift_enhance_range(magnitude):
    if False:
        print('Hello World!')
    'The `enhance` operations (brightness, contrast, color, sharpness) accept magnitudes\n    from [0, 2] range. However, the neutral magnitude is not 0 but 1 and the intuitive strength\n    of the operation increases the further the magnitude is from 1. So, we specify magnitudes range\n    to be in [0, 1] range, expect it to be randomly negated and then shift it by 1'
    return 1 + magnitude

@augmentation(mag_range=(0, 0.9), randomly_negate=True, mag_to_param=shift_enhance_range)
def brightness(data, parameter):
    if False:
        i = 10
        return i + 15
    return fn.brightness(data, brightness=parameter)

@augmentation(mag_range=(0, 0.9), randomly_negate=True, mag_to_param=shift_enhance_range)
def contrast(data, parameter):
    if False:
        while True:
            i = 10
    '\n    It follows PIL implementation of Contrast enhancement which uses a channel-weighted\n    mean as a contrast center.\n    '
    mean = fn.reductions.mean(data, axis_names='HW', keep_dims=True)
    rgb_weights = types.Constant(np.array([0.299, 0.587, 0.114], dtype=np.float32))
    center = fn.reductions.sum(mean * rgb_weights, axis_names='C', keep_dims=True)
    return fn.cast_like(center + (data - center) * parameter, data)

@augmentation(mag_range=(0, 0.9), randomly_negate=True, mag_to_param=shift_enhance_range)
def color(data, parameter):
    if False:
        while True:
            i = 10
    return fn.saturation(data, saturation=parameter)

def sharpness_kernel(magnitude):
    if False:
        while True:
            i = 10
    blur = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13
    ident = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    return -magnitude * blur + (1 + magnitude) * ident

def sharpness_kernel_shifted(magnitude):
    if False:
        return 10
    return sharpness_kernel(magnitude - 1)

@augmentation(mag_range=(0, 0.9), randomly_negate=True, mag_to_param=sharpness_kernel, param_device='auto')
def sharpness(data, kernel):
    if False:
        i = 10
        return i + 15
    "\n    The outputs correspond to PIL's ImageEnhance.Sharpness with the exception for 1px\n    border around the output. PIL computes convolution with smoothing filter only for\n    valid positions (no out-of-bounds filter positions) and pads the output with the input.\n    "
    return fn.experimental.filter(data, kernel)

def poster_mask_uint8(magnitude):
    if False:
        i = 10
        return i + 15
    magnitude = np.round(magnitude).astype(np.uint32)
    if magnitude <= 0:
        magnitude = 1
    elif magnitude > 8:
        magnitude = 8
    nbits = np.round(8 - magnitude).astype(np.uint32)
    removal_mask = np.uint8(2) ** nbits - 1
    return np.array(np.uint8(255) ^ removal_mask, dtype=np.uint8)

@augmentation(mag_range=(0, 4), mag_to_param=poster_mask_uint8, param_device='auto')
def posterize(data, mask):
    if False:
        print('Hello World!')
    return data & mask

@augmentation(mag_range=(256, 0), param_device='auto')
def solarize(data, threshold):
    if False:
        i = 10
        return i + 15
    sample_inv = types.Constant(255, dtype=types.UINT8) - data
    mask_unchanged = data < threshold
    mask_inverted = mask_unchanged ^ True
    return mask_unchanged * data + mask_inverted * sample_inv

def solarize_add_shift(shift):
    if False:
        while True:
            i = 10
    if shift >= 128:
        raise Exception('The solarize_add augmentation accepts shifts from 0 to 128')
    return np.uint8(shift)

@augmentation(mag_range=(0, 110), param_device='auto', mag_to_param=solarize_add_shift)
def solarize_add(data, shift):
    if False:
        for i in range(10):
            print('nop')
    mask_shifted = data < types.Constant(128, dtype=types.UINT8)
    mask_id = mask_shifted ^ True
    sample_shifted = data + shift
    return mask_shifted * sample_shifted + mask_id * data

@augmentation
def invert(data, _):
    if False:
        while True:
            i = 10
    return types.Constant(255, dtype=types.UINT8) - data

@augmentation
def equalize(data, _):
    if False:
        return 10
    "\n    DALI's equalize follows OpenCV's histogram equalization.\n    The PIL uses slightly different formula when transforming histogram's\n    cumulative sum into lookup table.\n    "
    return fn.experimental.equalize(data)

@augmentation
def auto_contrast(data, _):
    if False:
        print('Hello World!')
    lo = fn.reductions.min(data, axis_names='HW', keep_dims=True)
    hi = fn.reductions.max(data, axis_names='HW', keep_dims=True)
    diff = hi - lo
    mask_scale = diff > 0
    mask_id = mask_scale ^ True
    div_by = diff * mask_scale + types.Constant(255, dtype=types.UINT8) * mask_id
    scale = 255 / div_by
    scaled = (data - lo * mask_scale) * scale
    return fn.cast_like(scaled, data)

@augmentation
def identity(data, _):
    if False:
        i = 10
        return i + 15
    return data