from tifffile import imread as tifffile_imread
from tifffile import imwrite as tifffile_imwrite
__all__ = ['imread', 'imsave']

def imsave(fname, arr, **kwargs):
    if False:
        return 10
    "Load a tiff image to file.\n\n    Parameters\n    ----------\n    fname : str or file\n        File name or file-like object.\n    arr : ndarray\n        The array to write.\n    kwargs : keyword pairs, optional\n        Additional keyword arguments to pass through (see ``tifffile``'s\n        ``imwrite`` function).\n\n    Notes\n    -----\n    Provided by the tifffile library [1]_, and supports many\n    advanced image types including multi-page and floating-point.\n\n    This implementation will set ``photometric='RGB'`` when writing if the first\n    or last axis of `arr` has length 3 or 4. To override this, explicitly\n    pass the ``photometric`` kwarg.\n\n    This implementation will set ``planarconfig='SEPARATE'`` when writing if the\n    first axis of arr has length 3 or 4. To override this, explicitly\n    specify the ``planarconfig`` kwarg.\n\n    References\n    ----------\n    .. [1] https://pypi.org/project/tifffile/\n\n    "
    if arr.shape[0] in [3, 4]:
        if 'planarconfig' not in kwargs:
            kwargs['planarconfig'] = 'SEPARATE'
        rgb = True
    else:
        rgb = arr.shape[-1] in [3, 4]
    if rgb and 'photometric' not in kwargs:
        kwargs['photometric'] = 'RGB'
    return tifffile_imwrite(fname, arr, **kwargs)

def imread(fname, **kwargs):
    if False:
        print('Hello World!')
    "Load a tiff image from file.\n\n    Parameters\n    ----------\n    fname : str or file\n        File name or file-like-object.\n    kwargs : keyword pairs, optional\n        Additional keyword arguments to pass through (see ``tifffile``'s\n        ``imread`` function).\n\n    Notes\n    -----\n    Provided by the tifffile library [1]_, and supports many\n    advanced image types including multi-page and floating point.\n\n    References\n    ----------\n    .. [1] https://pypi.org/project/tifffile/\n\n    "
    if 'img_num' in kwargs:
        kwargs['key'] = kwargs.pop('img_num')
    return tifffile_imread(fname, **kwargs)