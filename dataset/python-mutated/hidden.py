"""Simple library to work out if a file is hidden on different platforms."""
import ctypes
import os
import stat
import sys
import beets.util

def _is_hidden_osx(path):
    if False:
        print('Hello World!')
    'Return whether or not a file is hidden on OS X.\n\n    This uses os.lstat to work out if a file has the "hidden" flag.\n    '
    file_stat = os.lstat(beets.util.syspath(path))
    if hasattr(file_stat, 'st_flags') and hasattr(stat, 'UF_HIDDEN'):
        return bool(file_stat.st_flags & stat.UF_HIDDEN)
    else:
        return False

def _is_hidden_win(path):
    if False:
        print('Hello World!')
    'Return whether or not a file is hidden on Windows.\n\n    This uses GetFileAttributes to work out if a file has the "hidden" flag\n    (FILE_ATTRIBUTE_HIDDEN).\n    '
    hidden_mask = 2
    attrs = ctypes.windll.kernel32.GetFileAttributesW(beets.util.syspath(path))
    return attrs >= 0 and attrs & hidden_mask

def _is_hidden_dot(path):
    if False:
        print('Hello World!')
    'Return whether or not a file starts with a dot.\n\n    Files starting with a dot are seen as "hidden" files on Unix-based OSes.\n    '
    return os.path.basename(path).startswith(b'.')

def is_hidden(path):
    if False:
        while True:
            i = 10
    'Return whether or not a file is hidden. `path` should be a\n    bytestring filename.\n\n    This method works differently depending on the platform it is called on.\n\n    On OS X, it uses both the result of `is_hidden_osx` and `is_hidden_dot` to\n    work out if a file is hidden.\n\n    On Windows, it uses the result of `is_hidden_win` to work out if a file is\n    hidden.\n\n    On any other operating systems (i.e. Linux), it uses `is_hidden_dot` to\n    work out if a file is hidden.\n    '
    if sys.platform == 'darwin':
        return _is_hidden_osx(path) or _is_hidden_dot(path)
    elif sys.platform == 'win32':
        return _is_hidden_win(path)
    else:
        return _is_hidden_dot(path)
__all__ = ['is_hidden']