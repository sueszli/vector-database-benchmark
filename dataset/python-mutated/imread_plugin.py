__all__ = ['imread', 'imsave']
from ...util.dtype import _convert
try:
    import imread as _imread
except ImportError:
    raise ImportError('Imread could not be foundPlease refer to http://pypi.python.org/pypi/imread/ for further instructions.')

def imread(fname, dtype=None):
    if False:
        i = 10
        return i + 15
    'Load an image from file.\n\n    Parameters\n    ----------\n    fname : str\n        Name of input file\n\n    '
    im = _imread.imread(fname)
    if dtype is not None:
        im = _convert(im, dtype)
    return im

def imsave(fname, arr, format_str=None):
    if False:
        for i in range(10):
            print('nop')
    'Save an image to disk.\n\n    Parameters\n    ----------\n    fname : str\n        Name of destination file.\n    arr : ndarray of uint8 or uint16\n        Array (image) to save.\n    format_str : str,optional\n        Format to save as.\n\n    Notes\n    -----\n    Currently, only 8-bit precision is supported.\n    '
    return _imread.imsave(fname, arr, formatstr=format_str)