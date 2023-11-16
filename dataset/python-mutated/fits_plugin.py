__all__ = ['imread', 'imread_collection']
import skimage.io as io
try:
    from astropy.io import fits
except ImportError:
    raise ImportError('Astropy could not be found. It is needed to read FITS files.\nPlease refer to https://www.astropy.org for installation\ninstructions.')

def imread(fname):
    if False:
        print('Hello World!')
    'Load an image from a FITS file.\n\n    Parameters\n    ----------\n    fname : string\n        Image file name, e.g. ``test.fits``.\n\n    Returns\n    -------\n    img_array : ndarray\n        Unlike plugins such as PIL, where different color bands/channels are\n        stored in the third dimension, FITS images are grayscale-only and can\n        be N-dimensional, so an array of the native FITS dimensionality is\n        returned, without color channels.\n\n        Currently if no image is found in the file, None will be returned\n\n    Notes\n    -----\n    Currently FITS ``imread()`` always returns the first image extension when\n    given a Multi-Extension FITS file; use ``imread_collection()`` (which does\n    lazy loading) to get all the extensions at once.\n\n    '
    with fits.open(fname) as hdulist:
        img_array = None
        for hdu in hdulist:
            if isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):
                if hdu.data is not None:
                    img_array = hdu.data
                    break
    return img_array

def imread_collection(load_pattern, conserve_memory=True):
    if False:
        i = 10
        return i + 15
    'Load a collection of images from one or more FITS files\n\n    Parameters\n    ----------\n    load_pattern : str or list\n        List of extensions to load. Filename globbing is currently\n        unsupported.\n    conserve_memory : bool\n        If True, never keep more than one in memory at a specific\n        time. Otherwise, images will be cached once they are loaded.\n\n    Returns\n    -------\n    ic : ImageCollection\n        Collection of images.\n\n    '
    intype = type(load_pattern)
    if intype is not list and intype is not str:
        raise TypeError('Input must be a filename or list of filenames')
    if intype is not list:
        load_pattern = [load_pattern]
    ext_list = []
    for filename in load_pattern:
        with fits.open(filename) as hdulist:
            for (n, hdu) in zip(range(len(hdulist)), hdulist):
                if isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):
                    try:
                        data_size = hdu.size
                    except TypeError:
                        data_size = hdu.size()
                    if data_size > 0:
                        ext_list.append((filename, n))
    return io.ImageCollection(ext_list, load_func=FITSFactory, conserve_memory=conserve_memory)

def FITSFactory(image_ext):
    if False:
        for i in range(10):
            print('nop')
    'Load an image extension from a FITS file and return a NumPy array\n\n    Parameters\n    ----------\n    image_ext : tuple\n        FITS extension to load, in the format ``(filename, ext_num)``.\n        The FITS ``(extname, extver)`` format is unsupported, since this\n        function is not called directly by the user and\n        ``imread_collection()`` does the work of figuring out which\n        extensions need loading.\n\n    '
    if not isinstance(image_ext, tuple):
        raise TypeError('Expected a tuple')
    if len(image_ext) != 2:
        raise ValueError('Expected a tuple of length 2')
    filename = image_ext[0]
    extnum = image_ext[1]
    if not (isinstance(filename, str) and isinstance(extnum, int)):
        raise ValueError('Expected a (filename, extension) tuple')
    with fits.open(filename) as hdulist:
        data = hdulist[extnum].data
    if data is None:
        raise RuntimeError(f'Extension {extnum} of {filename} has no data')
    return data