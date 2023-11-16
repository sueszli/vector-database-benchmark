from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
_TMP_COL_PREP_IMAGE = '_prepared_image'
_TMP_COL_RANDOM_ORDER = '_random_order'

def _resize_if_too_large(image, max_shape):
    if False:
        i = 10
        return i + 15
    width_f = image.width / max_shape[1]
    height_f = image.height / max_shape[0]
    f = max(width_f, height_f)
    if f > 1.0:
        (width, height) = (int(image.width / f), int(image.height / f))
    else:
        (width, height) = (image.width, image.height)
    width = min(width, max_shape[1])
    height = min(height, max_shape[0])
    return _tc.image_analysis.resize(image, width, height, 3, decode=True, resample='bilinear')

def _stretch_resize(image, shape):
    if False:
        i = 10
        return i + 15
    (height, width) = shape
    return _tc.image_analysis.resize(image, width, height, 3, decode=True, resample='bilinear')