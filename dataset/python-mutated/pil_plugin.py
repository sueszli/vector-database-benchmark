__all__ = ['imread', 'imsave']
import numpy as np
from PIL import Image
from ...util import img_as_ubyte, img_as_uint

def imread(fname, dtype=None, img_num=None, **kwargs):
    if False:
        while True:
            i = 10
    'Load an image from file.\n\n    Parameters\n    ----------\n    fname : str or file\n        File name or file-like-object.\n    dtype : numpy dtype object or string specifier\n        Specifies data type of array elements.\n    img_num : int, optional\n        Specifies which image to read in a file with multiple images\n        (zero-indexed).\n    kwargs : keyword pairs, optional\n        Addition keyword arguments to pass through.\n\n    Notes\n    -----\n    Files are read using the Python Imaging Library.\n    See PIL docs [1]_ for a list of supported formats.\n\n    References\n    ----------\n    .. [1] http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html\n    '
    if isinstance(fname, str):
        with open(fname, 'rb') as f:
            im = Image.open(f)
            return pil_to_ndarray(im, dtype=dtype, img_num=img_num)
    else:
        im = Image.open(fname)
        return pil_to_ndarray(im, dtype=dtype, img_num=img_num)

def pil_to_ndarray(image, dtype=None, img_num=None):
    if False:
        print('Hello World!')
    'Import a PIL Image object to an ndarray, in memory.\n\n    Parameters\n    ----------\n    Refer to ``imread``.\n\n    '
    try:
        image.getdata()[0]
    except OSError as e:
        site = 'http://pillow.readthedocs.org/en/latest/installation.html#external-libraries'
        pillow_error_message = str(e)
        error_message = f"Could not load '{image.filename}' \nReason: '{pillow_error_message}'\nPlease see documentation at: {site}"
        raise ValueError(error_message)
    frames = []
    grayscale = None
    i = 0
    while 1:
        try:
            image.seek(i)
        except EOFError:
            break
        frame = image
        if img_num is not None and img_num != i:
            image.getdata()[0]
            i += 1
            continue
        if image.format == 'PNG' and image.mode == 'I' and (dtype is None):
            dtype = 'uint16'
        if image.mode == 'P':
            if grayscale is None:
                grayscale = _palette_is_grayscale(image)
            if grayscale:
                frame = image.convert('L')
            elif image.format == 'PNG' and 'transparency' in image.info:
                frame = image.convert('RGBA')
            else:
                frame = image.convert('RGB')
        elif image.mode == '1':
            frame = image.convert('L')
        elif 'A' in image.mode:
            frame = image.convert('RGBA')
        elif image.mode == 'CMYK':
            frame = image.convert('RGB')
        if image.mode.startswith('I;16'):
            shape = image.size
            dtype = '>u2' if image.mode.endswith('B') else '<u2'
            if 'S' in image.mode:
                dtype = dtype.replace('u', 'i')
            frame = np.fromstring(frame.tobytes(), dtype)
            frame.shape = shape[::-1]
        else:
            frame = np.array(frame, dtype=dtype)
        frames.append(frame)
        i += 1
        if img_num is not None:
            break
    if hasattr(image, 'fp') and image.fp:
        image.fp.close()
    if img_num is None and len(frames) > 1:
        return np.array(frames)
    elif frames:
        return frames[0]
    elif img_num:
        raise IndexError(f'Could not find image  #{img_num}')

def _palette_is_grayscale(pil_image):
    if False:
        return 10
    'Return True if PIL image in palette mode is grayscale.\n\n    Parameters\n    ----------\n    pil_image : PIL image\n        PIL Image that is in Palette mode.\n\n    Returns\n    -------\n    is_grayscale : bool\n        True if all colors in image palette are gray.\n    '
    if pil_image.mode != 'P':
        raise ValueError('pil_image.mode must be equal to "P".')
    palette = np.asarray(pil_image.getpalette()).reshape((-1, 3))
    (start, stop) = pil_image.getextrema()
    valid_palette = palette[start:stop + 1]
    return np.allclose(np.diff(valid_palette), 0)

def ndarray_to_pil(arr, format_str=None):
    if False:
        i = 10
        return i + 15
    'Export an ndarray to a PIL object.\n\n    Parameters\n    ----------\n    Refer to ``imsave``.\n\n    '
    if arr.ndim == 3:
        arr = img_as_ubyte(arr)
        mode = {3: 'RGB', 4: 'RGBA'}[arr.shape[2]]
    elif format_str in ['png', 'PNG']:
        mode = 'I;16'
        mode_base = 'I'
        if arr.dtype.kind == 'f':
            arr = img_as_uint(arr)
        elif arr.max() < 256 and arr.min() >= 0:
            arr = arr.astype(np.uint8)
            mode = mode_base = 'L'
        else:
            arr = img_as_uint(arr)
    else:
        arr = img_as_ubyte(arr)
        mode = 'L'
        mode_base = 'L'
    try:
        array_buffer = arr.tobytes()
    except AttributeError:
        array_buffer = arr.tostring()
    if arr.ndim == 2:
        im = Image.new(mode_base, arr.T.shape)
        try:
            im.frombytes(array_buffer, 'raw', mode)
        except AttributeError:
            im.fromstring(array_buffer, 'raw', mode)
    else:
        image_shape = (arr.shape[1], arr.shape[0])
        try:
            im = Image.frombytes(mode, image_shape, array_buffer)
        except AttributeError:
            im = Image.fromstring(mode, image_shape, array_buffer)
    return im

def imsave(fname, arr, format_str=None, **kwargs):
    if False:
        while True:
            i = 10
    "Save an image to disk.\n\n    Parameters\n    ----------\n    fname : str or file-like object\n        Name of destination file.\n    arr : ndarray of uint8 or float\n        Array (image) to save.  Arrays of data-type uint8 should have\n        values in [0, 255], whereas floating-point arrays must be\n        in [0, 1].\n    format_str: str\n        Format to save as, this is defaulted to PNG if using a file-like\n        object; this will be derived from the extension if fname is a string\n    kwargs: dict\n        Keyword arguments to the Pillow save function (or tifffile save\n        function, for Tiff files). These are format dependent. For example,\n        Pillow's JPEG save function supports an integer ``quality`` argument\n        with values in [1, 95], while TIFFFile supports a ``compress``\n        integer argument with values in [0, 9].\n\n    Notes\n    -----\n    Use the Python Imaging Library.\n    See PIL docs [1]_ for a list of other supported formats.\n    All images besides single channel PNGs are converted using `img_as_uint8`.\n    Single Channel PNGs have the following behavior:\n    - Integer values in [0, 255] and Boolean types -> img_as_uint8\n    - Floating point and other integers -> img_as_uint16\n\n    References\n    ----------\n    .. [1] http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html\n    "
    if not isinstance(fname, str) and format_str is None:
        format_str = 'PNG'
    if isinstance(fname, str) and fname.lower().endswith('.png'):
        format_str = 'PNG'
    arr = np.asanyarray(arr)
    if arr.dtype.kind == 'b':
        arr = arr.astype(np.uint8)
    if arr.ndim not in (2, 3):
        raise ValueError(f'Invalid shape for image array: {arr.shape}')
    if arr.ndim == 3:
        if arr.shape[2] not in (3, 4):
            raise ValueError('Invalid number of channels in image array.')
    img = ndarray_to_pil(arr, format_str=format_str)
    img.save(fname, format=format_str, **kwargs)