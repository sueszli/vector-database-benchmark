"""
Module like __future__ for things that are changed between Python2 and Python3.

These are here to provide compatible fall-backs. This is required to run the
same code easily with both CPython2 and CPython3. Sometimes, we do not care
about the actual types, or API, but would rather just check for something to
be a "in (str, unicode)" rather than making useless version checks.

"""
import pkgutil
import sys
from hashlib import md5 as _md5
if str is bytes:
    import __builtin__ as builtins
else:
    import builtins
if str is bytes:
    long = long
else:
    long = int
if str is bytes:
    unicode = unicode
else:
    unicode = str
if str is bytes:

    def iterItems(d):
        if False:
            return 10
        return d.iteritems()
else:

    def iterItems(d):
        if False:
            print('Hello World!')
        return d.items()
if str is not bytes:
    raw_input = input
    xrange = range
    basestring = str
else:
    raw_input = raw_input
    xrange = xrange
    basestring = basestring
if str is bytes:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve
if str is bytes:
    from cStringIO import StringIO
else:
    from io import StringIO
if str is bytes:
    BytesIO = StringIO
else:
    from io import BytesIO
try:
    from functools import total_ordering
except ImportError:

    def total_ordering(cls):
        if False:
            while True:
                i = 10
        cls.__ne__ = lambda self, other: not self == other
        cls.__le__ = lambda self, other: self == other or self < other
        cls.__gt__ = lambda self, other: self != other and (not self < other)
        cls.__ge__ = lambda self, other: self == other and (not self < other)
        return cls
if str is bytes:
    from collections import Iterable, MutableSet
else:
    from collections.abc import Iterable, MutableSet
if str is bytes:
    intern = intern
else:
    intern = sys.intern
if str is bytes:
    to_byte = chr
    from_byte = ord
else:

    def to_byte(value):
        if False:
            return 10
        assert type(value) is int and 0 <= value < 256, value
        return bytes((value,))

    def from_byte(value):
        if False:
            print('Hello World!')
        assert type(value) is bytes and len(value) == 1, value
        return value[0]
try:
    from typing import GenericAlias
except ImportError:
    GenericAlias = None
try:
    from types import UnionType
except ImportError:
    UnionType = None
if str is bytes:
    try:
        import subprocess32 as subprocess
    except ImportError:
        import subprocess
else:
    import subprocess
WindowsError = OSError
PermissionError = PermissionError if str is not bytes else OSError
if not hasattr(pkgutil, 'ModuleInfo'):
    from collections import namedtuple
    ModuleInfo = namedtuple('ModuleInfo', 'module_finder name ispkg')

    def iter_modules(path=None, prefix=''):
        if False:
            print('Hello World!')
        for item in pkgutil.iter_modules(path, prefix):
            yield ModuleInfo(*item)
else:
    iter_modules = pkgutil.iter_modules
try:
    ExceptionGroup = ExceptionGroup
except NameError:
    ExceptionGroup = None
try:
    BaseExceptionGroup = BaseExceptionGroup
except NameError:
    BaseExceptionGroup = None
try:
    _md5()
except ValueError:

    def md5(value=b''):
        if False:
            i = 10
            return i + 15
        return _md5(value, usedforsecurity=False)
else:
    md5 = _md5
assert long
assert unicode
assert urlretrieve
assert StringIO
assert BytesIO
assert type(xrange) is type, xrange
assert total_ordering
assert intern
assert builtins
assert Iterable
assert MutableSet
assert subprocess
assert GenericAlias or intern
assert UnionType or intern