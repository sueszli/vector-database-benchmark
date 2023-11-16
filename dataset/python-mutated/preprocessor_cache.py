"""Records previous preprocessing operations and allows them to be repeated.

Used with object_detection.core.preprocessor. Passing a PreprocessorCache
into individual data augmentation functions or the general preprocess() function
will store all randomly generated variables in the PreprocessorCache. When
a preprocessor function is called multiple times with the same
PreprocessorCache object, that function will perform the same augmentation
on all calls.
"""
from collections import defaultdict

class PreprocessorCache(object):
    """Dictionary wrapper storing random variables generated during preprocessing.
  """
    ROTATION90 = 'rotation90'
    HORIZONTAL_FLIP = 'horizontal_flip'
    VERTICAL_FLIP = 'vertical_flip'
    PIXEL_VALUE_SCALE = 'pixel_value_scale'
    IMAGE_SCALE = 'image_scale'
    RGB_TO_GRAY = 'rgb_to_gray'
    ADJUST_BRIGHTNESS = 'adjust_brightness'
    ADJUST_CONTRAST = 'adjust_contrast'
    ADJUST_HUE = 'adjust_hue'
    ADJUST_SATURATION = 'adjust_saturation'
    DISTORT_COLOR = 'distort_color'
    STRICT_CROP_IMAGE = 'strict_crop_image'
    CROP_IMAGE = 'crop_image'
    PAD_IMAGE = 'pad_image'
    CROP_TO_ASPECT_RATIO = 'crop_to_aspect_ratio'
    RESIZE_METHOD = 'resize_method'
    PAD_TO_ASPECT_RATIO = 'pad_to_aspect_ratio'
    BLACK_PATCHES = 'black_patches'
    ADD_BLACK_PATCH = 'add_black_patch'
    SELECTOR = 'selector'
    SELECTOR_TUPLES = 'selector_tuples'
    SELF_CONCAT_IMAGE = 'self_concat_image'
    SSD_CROP_SELECTOR_ID = 'ssd_crop_selector_id'
    SSD_CROP_PAD_SELECTOR_ID = 'ssd_crop_pad_selector_id'
    JPEG_QUALITY = 'jpeg_quality'
    DOWNSCALE_TO_TARGET_PIXELS = 'downscale_to_target_pixels'
    PATCH_GAUSSIAN = 'patch_gaussian'
    _VALID_FNS = [ROTATION90, HORIZONTAL_FLIP, VERTICAL_FLIP, PIXEL_VALUE_SCALE, IMAGE_SCALE, RGB_TO_GRAY, ADJUST_BRIGHTNESS, ADJUST_CONTRAST, ADJUST_HUE, ADJUST_SATURATION, DISTORT_COLOR, STRICT_CROP_IMAGE, CROP_IMAGE, PAD_IMAGE, CROP_TO_ASPECT_RATIO, RESIZE_METHOD, PAD_TO_ASPECT_RATIO, BLACK_PATCHES, ADD_BLACK_PATCH, SELECTOR, SELECTOR_TUPLES, SELF_CONCAT_IMAGE, SSD_CROP_SELECTOR_ID, SSD_CROP_PAD_SELECTOR_ID, JPEG_QUALITY, DOWNSCALE_TO_TARGET_PIXELS, PATCH_GAUSSIAN]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._history = defaultdict(dict)

    def clear(self):
        if False:
            while True:
                i = 10
        'Resets cache.'
        self._history = defaultdict(dict)

    def get(self, function_id, key):
        if False:
            while True:
                i = 10
        'Gets stored value given a function id and key.\n\n    Args:\n      function_id: identifier for the preprocessing function used.\n      key: identifier for the variable stored.\n    Returns:\n      value: the corresponding value, expected to be a tensor or\n             nested structure of tensors.\n    Raises:\n      ValueError: if function_id is not one of the 23 valid function ids.\n    '
        if function_id not in self._VALID_FNS:
            raise ValueError('Function id not recognized: %s.' % str(function_id))
        return self._history[function_id].get(key)

    def update(self, function_id, key, value):
        if False:
            for i in range(10):
                print('nop')
        'Adds a value to the dictionary.\n\n    Args:\n      function_id: identifier for the preprocessing function used.\n      key: identifier for the variable stored.\n      value: the value to store, expected to be a tensor or nested structure\n             of tensors.\n    Raises:\n      ValueError: if function_id is not one of the 23 valid function ids.\n    '
        if function_id not in self._VALID_FNS:
            raise ValueError('Function id not recognized: %s.' % str(function_id))
        self._history[function_id][key] = value