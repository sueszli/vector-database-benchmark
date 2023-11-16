"""
This module contains simple input/output related functionality that is not
part of a larger framework or standard.
"""
import pickle
from astropy.utils.decorators import deprecated
__all__ = ['fnpickle', 'fnunpickle']

@deprecated(since='6.0', message='Use pickle from standard library, if you must')
def fnunpickle(fileorname, number=0):
    if False:
        print('Hello World!')
    'Unpickle pickled objects from a specified file and return the contents.\n\n    .. warning:: The ``pickle`` module is not secure. Only unpickle data you trust.\n\n    Parameters\n    ----------\n    fileorname : str or file-like\n        The file name or file from which to unpickle objects. If a file object,\n        it should have been opened in binary mode.\n    number : int\n        If 0, a single object will be returned (the first in the file). If >0,\n        this specifies the number of objects to be unpickled, and a list will\n        be returned with exactly that many objects. If <0, all objects in the\n        file will be unpickled and returned as a list.\n\n    Raises\n    ------\n    EOFError\n        If ``number`` is >0 and there are fewer than ``number`` objects in the\n        pickled file.\n\n    Returns\n    -------\n    contents : object or list\n        If ``number`` is 0, this is a individual object - the first one\n        unpickled from the file. Otherwise, it is a list of objects unpickled\n        from the file.\n\n    '
    if isinstance(fileorname, str):
        f = open(fileorname, 'rb')
        close = True
    else:
        f = fileorname
        close = False
    try:
        if number > 0:
            res = []
            for i in range(number):
                res.append(pickle.load(f))
        elif number < 0:
            res = []
            eof = False
            while not eof:
                try:
                    res.append(pickle.load(f))
                except EOFError:
                    eof = True
        else:
            res = pickle.load(f)
    finally:
        if close:
            f.close()
    return res

@deprecated(since='6.0', message='Use pickle from standard library, if you must')
def fnpickle(object, fileorname, protocol=None, append=False):
    if False:
        i = 10
        return i + 15
    'Pickle an object to a specified file.\n\n    Parameters\n    ----------\n    object\n        The python object to pickle.\n    fileorname : str or file-like\n        The filename or file into which the `object` should be pickled. If a\n        file object, it should have been opened in binary mode.\n    protocol : int or None\n        Pickle protocol to use - see the :mod:`pickle` module for details on\n        these options. If None, the most recent protocol will be used.\n    append : bool\n        If True, the object is appended to the end of the file, otherwise the\n        file will be overwritten (if a file object is given instead of a\n        file name, this has no effect).\n\n    '
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    if isinstance(fileorname, str):
        f = open(fileorname, 'ab' if append else 'wb')
        close = True
    else:
        f = fileorname
        close = False
    try:
        pickle.dump(object, f, protocol=protocol)
    finally:
        if close:
            f.close()