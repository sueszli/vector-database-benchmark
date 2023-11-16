__all__ = ['imread', 'imsave']
from functools import wraps
import numpy as np
from imageio.v3 import imread as imageio_imread, imwrite as imsave

@wraps(imageio_imread)
def imread(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    out = np.asarray(imageio_imread(*args, **kwargs))
    if not out.flags['WRITEABLE']:
        out = out.copy()
    return out