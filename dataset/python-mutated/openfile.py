"""
Handle file opening for read/write
"""
from pathlib import Path
from numpy.lib._iotools import _is_string_like

class EmptyContextManager:
    """
    This class is needed to allow file-like object to be used as
    context manager, but without getting closed.
    """

    def __init__(self, obj):
        if False:
            i = 10
            return i + 15
        self._obj = obj

    def __enter__(self):
        if False:
            return 10
        'When entering, return the embedded object'
        return self._obj

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Do not hide anything'
        return False

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._obj, name)

def _open(fname, mode, encoding):
    if False:
        print('Hello World!')
    if fname.endswith('.gz'):
        import gzip
        return gzip.open(fname, mode, encoding=encoding)
    else:
        return open(fname, mode, encoding=encoding)

def get_file_obj(fname, mode='r', encoding=None):
    if False:
        i = 10
        return i + 15
    "\n    Light wrapper to handle strings, path objects and let files (anything else)\n    pass through.\n\n    It also handle '.gz' files.\n\n    Parameters\n    ----------\n    fname : str, path object or file-like object\n        File to open / forward\n    mode : str\n        Argument passed to the 'open' or 'gzip.open' function\n    encoding : str\n        For Python 3 only, specify the encoding of the file\n\n    Returns\n    -------\n    A file-like object that is always a context-manager. If the `fname` was\n    already a file-like object, the returned context manager *will not\n    close the file*.\n    "
    if _is_string_like(fname):
        fname = Path(fname)
    if isinstance(fname, Path):
        return fname.open(mode=mode, encoding=encoding)
    elif hasattr(fname, 'open'):
        return fname.open(mode=mode, encoding=encoding)
    try:
        return open(fname, mode, encoding=encoding)
    except TypeError:
        try:
            if 'r' in mode:
                fname.read
            if 'w' in mode or 'a' in mode:
                fname.write
        except AttributeError:
            raise ValueError('fname must be a string or a file-like object')
        return EmptyContextManager(fname)