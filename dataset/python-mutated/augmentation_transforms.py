"""Transforms used in the Augmentation Policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
IMAGE_SIZE = 32
MEANS = [0.49139968, 0.48215841, 0.44653091]
STDS = [0.24703223, 0.24348513, 0.26158784]
PARAMETER_MAX = 10

def random_flip(x):
    if False:
        print('Hello World!')
    'Flip the input x horizontally with 50% probability.'
    if np.random.rand(1)[0] > 0.5:
        return np.fliplr(x)
    return x

def zero_pad_and_crop(img, amount=4):
    if False:
        while True:
            i = 10
    'Zero pad by `amount` zero pixels on each side then take a random crop.\n\n  Args:\n    img: numpy image that will be zero padded and cropped.\n    amount: amount of zeros to pad `img` with horizontally and verically.\n\n  Returns:\n    The cropped zero padded img. The returned numpy array will be of the same\n    shape as `img`.\n  '
    padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2, img.shape[2]))
    padded_img[amount:img.shape[0] + amount, amount:img.shape[1] + amount, :] = img
    top = np.random.randint(low=0, high=2 * amount)
    left = np.random.randint(low=0, high=2 * amount)
    new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
    return new_img

def create_cutout_mask(img_height, img_width, num_channels, size):
    if False:
        for i in range(10):
            print('nop')
    'Creates a zero mask used for cutout of shape `img_height` x `img_width`.\n\n  Args:\n    img_height: Height of image cutout mask will be applied to.\n    img_width: Width of image cutout mask will be applied to.\n    num_channels: Number of channels in the image.\n    size: Size of the zeros mask.\n\n  Returns:\n    A mask of shape `img_height` x `img_width` with all ones except for a\n    square of zeros of shape `size` x `size`. This mask is meant to be\n    elementwise multiplied with the original image. Additionally returns\n    the `upper_coord` and `lower_coord` which specify where the cutout mask\n    will be applied.\n  '
    assert img_height == img_width
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0
    mask = np.ones((img_height, img_width, num_channels))
    zeros = np.zeros((mask_height, mask_width, num_channels))
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = zeros
    return (mask, upper_coord, lower_coord)

def cutout_numpy(img, size=16):
    if False:
        while True:
            i = 10
    'Apply cutout with mask of shape `size` x `size` to `img`.\n\n  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.\n  This operation applies a `size`x`size` mask of zeros to a random location\n  within `img`.\n\n  Args:\n    img: Numpy image that cutout will be applied to.\n    size: Height/width of the cutout mask that will be\n\n  Returns:\n    A numpy tensor that is the result of applying the cutout mask to `img`.\n  '
    (img_height, img_width, num_channels) = (img.shape[0], img.shape[1], img.shape[2])
    assert len(img.shape) == 3
    (mask, _, _) = create_cutout_mask(img_height, img_width, num_channels, size)
    return img * mask

def float_parameter(level, maxval):
    if False:
        return 10
    'Helper function to scale `val` between 0 and maxval .\n\n  Args:\n    level: Level of the operation that will be between [0, `PARAMETER_MAX`].\n    maxval: Maximum value that the operation can have. This will be scaled\n      to level/PARAMETER_MAX.\n\n  Returns:\n    A float that results from scaling `maxval` according to `level`.\n  '
    return float(level) * maxval / PARAMETER_MAX

def int_parameter(level, maxval):
    if False:
        while True:
            i = 10
    'Helper function to scale `val` between 0 and maxval .\n\n  Args:\n    level: Level of the operation that will be between [0, `PARAMETER_MAX`].\n    maxval: Maximum value that the operation can have. This will be scaled\n      to level/PARAMETER_MAX.\n\n  Returns:\n    An int that results from scaling `maxval` according to `level`.\n  '
    return int(level * maxval / PARAMETER_MAX)

def pil_wrap(img):
    if False:
        i = 10
        return i + 15
    'Convert the `img` numpy tensor to a PIL Image.'
    return Image.fromarray(np.uint8((img * STDS + MEANS) * 255.0)).convert('RGBA')

def pil_unwrap(pil_img):
    if False:
        for i in range(10):
            print('nop')
    'Converts the PIL img to a numpy array.'
    pic_array = np.array(pil_img.getdata()).reshape((32, 32, 4)) / 255.0
    (i1, i2) = np.where(pic_array[:, :, 3] == 0)
    pic_array = (pic_array[:, :, :3] - MEANS) / STDS
    pic_array[i1, i2] = [0, 0, 0]
    return pic_array

def apply_policy(policy, img):
    if False:
        i = 10
        return i + 15
    'Apply the `policy` to the numpy `img`.\n\n  Args:\n    policy: A list of tuples with the form (name, probability, level) where\n      `name` is the name of the augmentation operation to apply, `probability`\n      is the probability of applying the operation and `level` is what strength\n      the operation to apply.\n    img: Numpy image that will have `policy` applied to it.\n\n  Returns:\n    The result of applying `policy` to `img`.\n  '
    pil_img = pil_wrap(img)
    for xform in policy:
        assert len(xform) == 3
        (name, probability, level) = xform
        xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level)
        pil_img = xform_fn(pil_img)
    return pil_unwrap(pil_img)

class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        if False:
            while True:
                i = 10
        self.f = func
        self.name = name

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        if False:
            print('Hello World!')
        return self.f(pil_img)

class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        if False:
            while True:
                i = 10
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level):
        if False:
            print('Hello World!')

        def return_function(im):
            if False:
                i = 10
                return i + 15
            if random.random() < probability:
                im = self.xform(im, level)
            return im
        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)

    def do_transform(self, image, level):
        if False:
            i = 10
            return i + 15
        f = self.pil_transformer(PARAMETER_MAX, level)
        return pil_unwrap(f(pil_wrap(image)))
identity = TransformT('identity', lambda pil_img, level: pil_img)
flip_lr = TransformT('FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT('FlipUD', lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
auto_contrast = TransformT('AutoContrast', lambda pil_img, level: ImageOps.autocontrast(pil_img.convert('RGB')).convert('RGBA'))
equalize = TransformT('Equalize', lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB')).convert('RGBA'))
invert = TransformT('Invert', lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB')).convert('RGBA'))
blur = TransformT('Blur', lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT('Smooth', lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))

def _rotate_impl(pil_img, level):
    if False:
        print('Hello World!')
    'Rotates `pil_img` from -30 to 30 degrees depending on `level`.'
    degrees = int_parameter(level, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)
rotate = TransformT('Rotate', _rotate_impl)

def _posterize_impl(pil_img, level):
    if False:
        return 10
    'Applies PIL Posterize to `pil_img`.'
    level = int_parameter(level, 4)
    return ImageOps.posterize(pil_img.convert('RGB'), 4 - level).convert('RGBA')
posterize = TransformT('Posterize', _posterize_impl)

def _shear_x_impl(pil_img, level):
    if False:
        return 10
    'Applies PIL ShearX to `pil_img`.\n\n  The ShearX operation shears the image along the horizontal axis with `level`\n  magnitude.\n\n  Args:\n    pil_img: Image in PIL object.\n    level: Strength of the operation specified as an Integer from\n      [0, `PARAMETER_MAX`].\n\n  Returns:\n    A PIL Image that has had ShearX applied to it.\n  '
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, level, 0, 0, 1, 0))
shear_x = TransformT('ShearX', _shear_x_impl)

def _shear_y_impl(pil_img, level):
    if False:
        for i in range(10):
            print('nop')
    'Applies PIL ShearY to `pil_img`.\n\n  The ShearY operation shears the image along the vertical axis with `level`\n  magnitude.\n\n  Args:\n    pil_img: Image in PIL object.\n    level: Strength of the operation specified as an Integer from\n      [0, `PARAMETER_MAX`].\n\n  Returns:\n    A PIL Image that has had ShearX applied to it.\n  '
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, level, 1, 0))
shear_y = TransformT('ShearY', _shear_y_impl)

def _translate_x_impl(pil_img, level):
    if False:
        for i in range(10):
            print('nop')
    'Applies PIL TranslateX to `pil_img`.\n\n  Translate the image in the horizontal direction by `level`\n  number of pixels.\n\n  Args:\n    pil_img: Image in PIL object.\n    level: Strength of the operation specified as an Integer from\n      [0, `PARAMETER_MAX`].\n\n  Returns:\n    A PIL Image that has had TranslateX applied to it.\n  '
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, level, 0, 1, 0))
translate_x = TransformT('TranslateX', _translate_x_impl)

def _translate_y_impl(pil_img, level):
    if False:
        return 10
    'Applies PIL TranslateY to `pil_img`.\n\n  Translate the image in the vertical direction by `level`\n  number of pixels.\n\n  Args:\n    pil_img: Image in PIL object.\n    level: Strength of the operation specified as an Integer from\n      [0, `PARAMETER_MAX`].\n\n  Returns:\n    A PIL Image that has had TranslateY applied to it.\n  '
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, 0, 1, level))
translate_y = TransformT('TranslateY', _translate_y_impl)

def _crop_impl(pil_img, level, interpolation=Image.BILINEAR):
    if False:
        for i in range(10):
            print('nop')
    'Applies a crop to `pil_img` with the size depending on the `level`.'
    cropped = pil_img.crop((level, level, IMAGE_SIZE - level, IMAGE_SIZE - level))
    resized = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), interpolation)
    return resized
crop_bilinear = TransformT('CropBilinear', _crop_impl)

def _solarize_impl(pil_img, level):
    if False:
        print('Hello World!')
    'Applies PIL Solarize to `pil_img`.\n\n  Translate the image in the vertical direction by `level`\n  number of pixels.\n\n  Args:\n    pil_img: Image in PIL object.\n    level: Strength of the operation specified as an Integer from\n      [0, `PARAMETER_MAX`].\n\n  Returns:\n    A PIL Image that has had Solarize applied to it.\n  '
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img.convert('RGB'), 256 - level).convert('RGBA')
solarize = TransformT('Solarize', _solarize_impl)

def _cutout_pil_impl(pil_img, level):
    if False:
        while True:
            i = 10
    'Apply cutout to pil_img at the specified level.'
    size = int_parameter(level, 20)
    if size <= 0:
        return pil_img
    (img_height, img_width, num_channels) = (32, 32, 3)
    (_, upper_coord, lower_coord) = create_cutout_mask(img_height, img_width, num_channels, size)
    pixels = pil_img.load()
    for i in range(upper_coord[0], lower_coord[0]):
        for j in range(upper_coord[1], lower_coord[1]):
            pixels[i, j] = (125, 122, 113, 0)
    return pil_img
cutout = TransformT('Cutout', _cutout_pil_impl)

def _enhancer_impl(enhancer):
    if False:
        for i in range(10):
            print('nop')
    'Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL.'

    def impl(pil_img, level):
        if False:
            while True:
                i = 10
        v = float_parameter(level, 1.8) + 0.1
        return enhancer(pil_img).enhance(v)
    return impl
color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
ALL_TRANSFORMS = [flip_lr, flip_ud, auto_contrast, equalize, invert, rotate, posterize, crop_bilinear, solarize, color, contrast, brightness, sharpness, shear_x, shear_y, translate_x, translate_y, cutout, blur, smooth]
NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}
TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()