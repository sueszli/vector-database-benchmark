import numpy as np
from .dtype import dtype_limits

def invert(image, signed_float=False):
    if False:
        while True:
            i = 10
    'Invert an image.\n\n    Invert the intensity range of the input image, so that the dtype maximum\n    is now the dtype minimum, and vice-versa. This operation is\n    slightly different depending on the input dtype:\n\n    - unsigned integers: subtract the image from the dtype maximum\n    - signed integers: subtract the image from -1 (see Notes)\n    - floats: subtract the image from 1 (if signed_float is False, so we\n      assume the image is unsigned), or from 0 (if signed_float is True).\n\n    See the examples for clarification.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    signed_float : bool, optional\n        If True and the image is of type float, the range is assumed to\n        be [-1, 1]. If False and the image is of type float, the range is\n        assumed to be [0, 1].\n\n    Returns\n    -------\n    inverted : ndarray\n        Inverted image.\n\n    Notes\n    -----\n    Ideally, for signed integers we would simply multiply by -1. However,\n    signed integer ranges are asymmetric. For example, for np.int8, the range\n    of possible values is [-128, 127], so that -128 * -1 equals -128! By\n    subtracting from -1, we correctly map the maximum dtype value to the\n    minimum.\n\n    Examples\n    --------\n    >>> img = np.array([[100,  0, 200],\n    ...                 [  0, 50,   0],\n    ...                 [ 30,  0, 255]], np.uint8)\n    >>> invert(img)\n    array([[155, 255,  55],\n           [255, 205, 255],\n           [225, 255,   0]], dtype=uint8)\n    >>> img2 = np.array([[ -2, 0, -128],\n    ...                  [127, 0,    5]], np.int8)\n    >>> invert(img2)\n    array([[   1,   -1,  127],\n           [-128,   -1,   -6]], dtype=int8)\n    >>> img3 = np.array([[ 0., 1., 0.5, 0.75]])\n    >>> invert(img3)\n    array([[1.  , 0.  , 0.5 , 0.25]])\n    >>> img4 = np.array([[ 0., 1., -1., -0.25]])\n    >>> invert(img4, signed_float=True)\n    array([[-0.  , -1.  ,  1.  ,  0.25]])\n    '
    if image.dtype == 'bool':
        inverted = ~image
    elif np.issubdtype(image.dtype, np.unsignedinteger):
        max_val = dtype_limits(image, clip_negative=False)[1]
        inverted = np.subtract(max_val, image, dtype=image.dtype)
    elif np.issubdtype(image.dtype, np.signedinteger):
        inverted = np.subtract(-1, image, dtype=image.dtype)
    elif signed_float:
        inverted = -image
    else:
        inverted = np.subtract(1, image, dtype=image.dtype)
    return inverted