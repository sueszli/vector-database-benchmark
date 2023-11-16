"""
Pure-Python implementation of a Python 3-like bytes object for Python 2.

Why do this? Without it, the Python 2 bytes object is a very, very
different beast to the Python 3 bytes object.
"""
from numbers import Integral
import string
import copy
from future.utils import istext, isbytes, PY2, PY3, with_metaclass
from future.types import no, issubset
from future.types.newobject import newobject
if PY2:
    from collections import Iterable
else:
    from collections.abc import Iterable
_builtin_bytes = bytes
if PY3:
    unicode = str

class BaseNewBytes(type):

    def __instancecheck__(cls, instance):
        if False:
            while True:
                i = 10
        if cls == newbytes:
            return isinstance(instance, _builtin_bytes)
        else:
            return issubclass(instance.__class__, cls)

def _newchr(x):
    if False:
        i = 10
        return i + 15
    if isinstance(x, str):
        return x.encode('ascii')
    else:
        return chr(x)

class newbytes(with_metaclass(BaseNewBytes, _builtin_bytes)):
    """
    A backport of the Python 3 bytes object to Py2
    """

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        From the Py3 bytes docstring:\n\n        bytes(iterable_of_ints) -> bytes\n        bytes(string, encoding[, errors]) -> bytes\n        bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer\n        bytes(int) -> bytes object of size given by the parameter initialized with null bytes\n        bytes() -> empty bytes object\n\n        Construct an immutable array of bytes from:\n          - an iterable yielding integers in range(256)\n          - a text string encoded using the specified encoding\n          - any object implementing the buffer API.\n          - an integer\n        '
        encoding = None
        errors = None
        if len(args) == 0:
            return super(newbytes, cls).__new__(cls)
        elif len(args) >= 2:
            args = list(args)
            if len(args) == 3:
                errors = args.pop()
            encoding = args.pop()
        if type(args[0]) == newbytes:
            return args[0]
        elif isinstance(args[0], _builtin_bytes):
            value = args[0]
        elif isinstance(args[0], unicode):
            try:
                if 'encoding' in kwargs:
                    assert encoding is None
                    encoding = kwargs['encoding']
                if 'errors' in kwargs:
                    assert errors is None
                    errors = kwargs['errors']
            except AssertionError:
                raise TypeError('Argument given by name and position')
            if encoding is None:
                raise TypeError('unicode string argument without an encoding')
            newargs = [encoding]
            if errors is not None:
                newargs.append(errors)
            value = args[0].encode(*newargs)
        elif hasattr(args[0], '__bytes__'):
            value = args[0].__bytes__()
        elif isinstance(args[0], Iterable):
            if len(args[0]) == 0:
                value = b''
            else:
                try:
                    value = bytearray([_newchr(x) for x in args[0]])
                except:
                    raise ValueError('bytes must be in range(0, 256)')
        elif isinstance(args[0], Integral):
            if args[0] < 0:
                raise ValueError('negative count')
            value = b'\x00' * args[0]
        else:
            value = args[0]
        if type(value) == newbytes:
            return copy.copy(value)
        else:
            return super(newbytes, cls).__new__(cls, value)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'b' + super(newbytes, self).__repr__()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'b' + "'{0}'".format(super(newbytes, self).__str__())

    def __getitem__(self, y):
        if False:
            print('Hello World!')
        value = super(newbytes, self).__getitem__(y)
        if isinstance(y, Integral):
            return ord(value)
        else:
            return newbytes(value)

    def __getslice__(self, *args):
        if False:
            return 10
        return self.__getitem__(slice(*args))

    def __contains__(self, key):
        if False:
            print('Hello World!')
        if isinstance(key, int):
            newbyteskey = newbytes([key])
        elif type(key) == newbytes:
            newbyteskey = key
        else:
            newbyteskey = newbytes(key)
        return issubset(list(newbyteskey), list(self))

    @no(unicode)
    def __add__(self, other):
        if False:
            return 10
        return newbytes(super(newbytes, self).__add__(other))

    @no(unicode)
    def __radd__(self, left):
        if False:
            i = 10
            return i + 15
        return newbytes(left) + self

    @no(unicode)
    def __mul__(self, other):
        if False:
            print('Hello World!')
        return newbytes(super(newbytes, self).__mul__(other))

    @no(unicode)
    def __rmul__(self, other):
        if False:
            print('Hello World!')
        return newbytes(super(newbytes, self).__rmul__(other))

    def __mod__(self, vals):
        if False:
            while True:
                i = 10
        if isinstance(vals, newbytes):
            vals = _builtin_bytes.__str__(vals)
        elif isinstance(vals, tuple):
            newvals = []
            for v in vals:
                if isinstance(v, newbytes):
                    v = _builtin_bytes.__str__(v)
                newvals.append(v)
            vals = tuple(newvals)
        elif hasattr(vals.__class__, '__getitem__') and hasattr(vals.__class__, 'iteritems'):
            for (k, v) in vals.iteritems():
                if isinstance(v, newbytes):
                    vals[k] = _builtin_bytes.__str__(v)
        return _builtin_bytes.__mod__(self, vals)

    def __imod__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__mod__(other)

    def join(self, iterable_of_bytes):
        if False:
            i = 10
            return i + 15
        errmsg = 'sequence item {0}: expected bytes, {1} found'
        if isbytes(iterable_of_bytes) or istext(iterable_of_bytes):
            raise TypeError(errmsg.format(0, type(iterable_of_bytes)))
        for (i, item) in enumerate(iterable_of_bytes):
            if istext(item):
                raise TypeError(errmsg.format(i, type(item)))
        return newbytes(super(newbytes, self).join(iterable_of_bytes))

    @classmethod
    def fromhex(cls, string):
        if False:
            i = 10
            return i + 15
        return cls(string.replace(' ', '').decode('hex'))

    @no(unicode)
    def find(self, sub, *args):
        if False:
            i = 10
            return i + 15
        return super(newbytes, self).find(sub, *args)

    @no(unicode)
    def rfind(self, sub, *args):
        if False:
            return 10
        return super(newbytes, self).rfind(sub, *args)

    @no(unicode, (1, 2))
    def replace(self, old, new, *args):
        if False:
            i = 10
            return i + 15
        return newbytes(super(newbytes, self).replace(old, new, *args))

    def encode(self, *args):
        if False:
            print('Hello World!')
        raise AttributeError('encode method has been disabled in newbytes')

    def decode(self, encoding='utf-8', errors='strict'):
        if False:
            print('Hello World!')
        "\n        Returns a newstr (i.e. unicode subclass)\n\n        Decode B using the codec registered for encoding. Default encoding\n        is 'utf-8'. errors may be given to set a different error\n        handling scheme.  Default is 'strict' meaning that encoding errors raise\n        a UnicodeDecodeError.  Other possible values are 'ignore' and 'replace'\n        as well as any other name registered with codecs.register_error that is\n        able to handle UnicodeDecodeErrors.\n        "
        from future.types.newstr import newstr
        if errors == 'surrogateescape':
            from future.utils.surrogateescape import register_surrogateescape
            register_surrogateescape()
        return newstr(super(newbytes, self).decode(encoding, errors))

    @no(unicode)
    def startswith(self, prefix, *args):
        if False:
            return 10
        return super(newbytes, self).startswith(prefix, *args)

    @no(unicode)
    def endswith(self, prefix, *args):
        if False:
            i = 10
            return i + 15
        return super(newbytes, self).endswith(prefix, *args)

    @no(unicode)
    def split(self, sep=None, maxsplit=-1):
        if False:
            for i in range(10):
                print('nop')
        parts = super(newbytes, self).split(sep, maxsplit)
        return [newbytes(part) for part in parts]

    def splitlines(self, keepends=False):
        if False:
            return 10
        '\n        B.splitlines([keepends]) -> list of lines\n\n        Return a list of the lines in B, breaking at line boundaries.\n        Line breaks are not included in the resulting list unless keepends\n        is given and true.\n        '
        parts = super(newbytes, self).splitlines(keepends)
        return [newbytes(part) for part in parts]

    @no(unicode)
    def rsplit(self, sep=None, maxsplit=-1):
        if False:
            print('Hello World!')
        parts = super(newbytes, self).rsplit(sep, maxsplit)
        return [newbytes(part) for part in parts]

    @no(unicode)
    def partition(self, sep):
        if False:
            i = 10
            return i + 15
        parts = super(newbytes, self).partition(sep)
        return tuple((newbytes(part) for part in parts))

    @no(unicode)
    def rpartition(self, sep):
        if False:
            i = 10
            return i + 15
        parts = super(newbytes, self).rpartition(sep)
        return tuple((newbytes(part) for part in parts))

    @no(unicode, (1,))
    def rindex(self, sub, *args):
        if False:
            i = 10
            return i + 15
        '\n        S.rindex(sub [,start [,end]]) -> int\n\n        Like S.rfind() but raise ValueError when the substring is not found.\n        '
        pos = self.rfind(sub, *args)
        if pos == -1:
            raise ValueError('substring not found')

    @no(unicode)
    def index(self, sub, *args):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns index of sub in bytes.\n        Raises ValueError if byte is not in bytes and TypeError if can't\n        be converted bytes or its length is not 1.\n        "
        if isinstance(sub, int):
            if len(args) == 0:
                (start, end) = (0, len(self))
            elif len(args) == 1:
                start = args[0]
            elif len(args) == 2:
                (start, end) = args
            else:
                raise TypeError('takes at most 3 arguments')
            return list(self)[start:end].index(sub)
        if not isinstance(sub, bytes):
            try:
                sub = self.__class__(sub)
            except (TypeError, ValueError):
                raise TypeError("can't convert sub to bytes")
        try:
            return super(newbytes, self).index(sub, *args)
        except ValueError:
            raise ValueError('substring not found')

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, (_builtin_bytes, bytearray)):
            return super(newbytes, self).__eq__(other)
        else:
            return False

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__ne__(other)
        else:
            return True
    unorderable_err = 'unorderable types: bytes() and {0}'

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__lt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __le__(self, other):
        if False:
            return 10
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__le__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __gt__(self, other):
        if False:
            return 10
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__gt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__ge__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __native__(self):
        if False:
            i = 10
            return i + 15
        return super(newbytes, self).__str__()

    def __getattribute__(self, name):
        if False:
            i = 10
            return i + 15
        "\n        A trick to cause the ``hasattr`` builtin-fn to return False for\n        the 'encode' method on Py2.\n        "
        if name in ['encode', u'encode']:
            raise AttributeError('encode method has been disabled in newbytes')
        return super(newbytes, self).__getattribute__(name)

    @no(unicode)
    def rstrip(self, bytes_to_strip=None):
        if False:
            i = 10
            return i + 15
        '\n        Strip trailing bytes contained in the argument.\n        If the argument is omitted, strip trailing ASCII whitespace.\n        '
        return newbytes(super(newbytes, self).rstrip(bytes_to_strip))

    @no(unicode)
    def strip(self, bytes_to_strip=None):
        if False:
            while True:
                i = 10
        '\n        Strip leading and trailing bytes contained in the argument.\n        If the argument is omitted, strip trailing ASCII whitespace.\n        '
        return newbytes(super(newbytes, self).strip(bytes_to_strip))

    def lower(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        b.lower() -> copy of b\n\n        Return a copy of b with all ASCII characters converted to lowercase.\n        '
        return newbytes(super(newbytes, self).lower())

    @no(unicode)
    def upper(self):
        if False:
            print('Hello World!')
        '\n        b.upper() -> copy of b\n\n        Return a copy of b with all ASCII characters converted to uppercase.\n        '
        return newbytes(super(newbytes, self).upper())

    @classmethod
    @no(unicode)
    def maketrans(cls, frm, to):
        if False:
            for i in range(10):
                print('nop')
        '\n        B.maketrans(frm, to) -> translation table\n\n        Return a translation table (a bytes object of length 256) suitable\n        for use in the bytes or bytearray translate method where each byte\n        in frm is mapped to the byte at the same position in to.\n        The bytes objects frm and to must be of the same length.\n        '
        return newbytes(string.maketrans(frm, to))
__all__ = ['newbytes']