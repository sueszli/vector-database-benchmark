import functools
import hashlib
import unittest
try:
    import _hashlib
except ImportError:
    _hashlib = None

def requires_hashdigest(digestname, openssl=None, usedforsecurity=True):
    if False:
        return 10
    "Decorator raising SkipTest if a hashing algorithm is not available\n\n    The hashing algorithm could be missing or blocked by a strict crypto\n    policy.\n\n    If 'openssl' is True, then the decorator checks that OpenSSL provides\n    the algorithm. Otherwise the check falls back to built-in\n    implementations. The usedforsecurity flag is passed to the constructor.\n\n    ValueError: [digital envelope routines: EVP_DigestInit_ex] disabled for FIPS\n    ValueError: unsupported hash type md4\n    "

    def decorator(func_or_class):
        if False:
            i = 10
            return i + 15
        if isinstance(func_or_class, type):
            setUpClass = func_or_class.__dict__.get('setUpClass')
            if setUpClass is None:

                def setUpClass(cls):
                    if False:
                        while True:
                            i = 10
                    super(func_or_class, cls).setUpClass()
                setUpClass.__qualname__ = func_or_class.__qualname__ + '.setUpClass'
                setUpClass.__module__ = func_or_class.__module__
            else:
                setUpClass = setUpClass.__func__
            setUpClass = classmethod(decorator(setUpClass))
            func_or_class.setUpClass = setUpClass
            return func_or_class

        @functools.wraps(func_or_class)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            try:
                if openssl and _hashlib is not None:
                    _hashlib.new(digestname, usedforsecurity=usedforsecurity)
                else:
                    hashlib.new(digestname, usedforsecurity=usedforsecurity)
            except ValueError:
                raise unittest.SkipTest(f"hash digest '{digestname}' is not available.")
            return func_or_class(*args, **kwargs)
        return wrapper
    return decorator