"""
Python 3.X compatibility tools.

While this file was originally intended for Python 2 -> 3 transition,
it is now used to create a compatibility layer between different
minor versions of Python 3.

While the active version of numpy may not support a given version of python, we
allow downstream libraries to continue to use these shims for forward
compatibility with numpy while they transition their code to newer versions of
Python.
"""
__all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar', 'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested', 'asstr', 'open_latin1', 'long', 'basestring', 'sixu', 'integer_types', 'is_pathlib_path', 'npy_load_module', 'Path', 'pickle', 'contextlib_nullcontext', 'os_fspath', 'os_PathLike']
import sys
import os
from pathlib import Path
import io
try:
    import pickle5 as pickle
except ImportError:
    import pickle
long = int
integer_types = (int,)
basestring = str
unicode = str
bytes = bytes

def asunicode(s):
    if False:
        while True:
            i = 10
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)

def asbytes(s):
    if False:
        i = 10
        return i + 15
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')

def asstr(s):
    if False:
        return 10
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)

def isfileobj(f):
    if False:
        print('Hello World!')
    if not isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter)):
        return False
    try:
        f.fileno()
        return True
    except OSError:
        return False

def open_latin1(filename, mode='r'):
    if False:
        for i in range(10):
            print('nop')
    return open(filename, mode=mode, encoding='iso-8859-1')

def sixu(s):
    if False:
        while True:
            i = 10
    return s
strchar = 'U'

def getexception():
    if False:
        for i in range(10):
            print('nop')
    return sys.exc_info()[1]

def asbytes_nested(x):
    if False:
        i = 10
        return i + 15
    if hasattr(x, '__iter__') and (not isinstance(x, (bytes, unicode))):
        return [asbytes_nested(y) for y in x]
    else:
        return asbytes(x)

def asunicode_nested(x):
    if False:
        return 10
    if hasattr(x, '__iter__') and (not isinstance(x, (bytes, unicode))):
        return [asunicode_nested(y) for y in x]
    else:
        return asunicode(x)

def is_pathlib_path(obj):
    if False:
        print('Hello World!')
    '\n    Check whether obj is a `pathlib.Path` object.\n\n    Prefer using ``isinstance(obj, os.PathLike)`` instead of this function.\n    '
    return isinstance(obj, Path)

class contextlib_nullcontext:
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True

    .. note::
        Prefer using `contextlib.nullcontext` instead of this context manager.
    """

    def __init__(self, enter_result=None):
        if False:
            i = 10
            return i + 15
        self.enter_result = enter_result

    def __enter__(self):
        if False:
            return 10
        return self.enter_result

    def __exit__(self, *excinfo):
        if False:
            while True:
                i = 10
        pass

def npy_load_module(name, fn, info=None):
    if False:
        print('Hello World!')
    '\n    Load a module. Uses ``load_module`` which will be deprecated in python\n    3.12. An alternative that uses ``exec_module`` is in\n    numpy.distutils.misc_util.exec_mod_from_location\n\n    .. versionadded:: 1.11.2\n\n    Parameters\n    ----------\n    name : str\n        Full module name.\n    fn : str\n        Path to module file.\n    info : tuple, optional\n        Only here for backward compatibility with Python 2.*.\n\n    Returns\n    -------\n    mod : module\n\n    '
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(name, fn).load_module()
os_fspath = os.fspath
os_PathLike = os.PathLike