""" Wrapper around appdirs from PyPI

We do not assume to be installed and fallback to an inline copy and if that
is not installed, we use our own code for best effort.
"""
from __future__ import absolute_import
import errno
import os
import tempfile
from nuitka.__past__ import PermissionError
from nuitka.Tracing import general
from .FileOperations import makePath
from .Importing import importFromInlineCopy
appdirs = importFromInlineCopy('appdirs', must_exist=False, delete_module=True)
if appdirs is None:
    import appdirs
_cache_dir = None

def getCacheDir():
    if False:
        return 10
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = os.getenv('NUITKA_CACHE_DIR')
        if _cache_dir:
            _cache_dir = os.path.expanduser(_cache_dir)
        elif appdirs is not None:
            _cache_dir = appdirs.user_cache_dir('Nuitka', None)
        else:
            _cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'Nuitka')
        if _cache_dir.startswith(('/nonexistent/', '/sbuild-nonexistent/', '/homeless-shelter/')):
            _cache_dir = os.path.join(tempfile.gettempdir(), 'Nuitka')
        try:
            makePath(_cache_dir)
        except PermissionError as e:
            if e.errno != errno.EACCES:
                raise
            general.sysexit("Error, failed to create cache directory '%s'. If this is due to a special environment, please consider making a PR for a general solution that adds support for it, or use 'NUITKA_CACHE_DIR' set to a writable directory." % _cache_dir)
    return _cache_dir