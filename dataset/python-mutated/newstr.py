"""
This module redefines ``str`` on Python 2.x to be a subclass of the Py2
``unicode`` type that behaves like the Python 3.x ``str``.

The main differences between ``newstr`` and Python 2.x's ``unicode`` type are
the stricter type-checking and absence of a `u''` prefix in the representation.

It is designed to be used together with the ``unicode_literals`` import
as follows:

    >>> from __future__ import unicode_literals
    >>> from builtins import str, isinstance

On Python 3.x and normally on Python 2.x, these expressions hold

    >>> str('blah') is 'blah'
    True
    >>> isinstance('blah', str)
    True

However, on Python 2.x, with this import:

    >>> from __future__ import unicode_literals

the same expressions are False:

    >>> str('blah') is 'blah'
    False
    >>> isinstance('blah', str)
    False

This module is designed to be imported together with ``unicode_literals`` on
Python 2 to bring the meaning of ``str`` back into alignment with unprefixed
string literals (i.e. ``unicode`` subclasses).

Note that ``str()`` (and ``print()``) would then normally call the
``__unicode__`` method on objects in Python 2. To define string
representations of your objects portably across Py3 and Py2, use the
:func:`python_2_unicode_compatible` decorator in  :mod:`future.utils`.

"""
from numbers import Number
from future.utils import PY3, istext, with_metaclass, isnewbytes
from future.types import no, issubset
from future.types.newobject import newobject
if PY3:
    unicode = str
    from collections.abc import Iterable
else:
    from collections import Iterable

class BaseNewStr(type):

    def __instancecheck__(cls, instance):
        if False:
            for i in range(10):
                print('nop')
        if cls == newstr:
            return isinstance(instance, unicode)
        else:
            return issubclass(instance.__class__, cls)

class newstr(with_metaclass(BaseNewStr, unicode)):
    """
    A backport of the Python 3 str object to Py2
    """
    no_convert_msg = "Can't convert '{0}' object to str implicitly"

    def __new__(cls, *args, **kwargs):
        if False:
            return 10
        "\n        From the Py3 str docstring:\n\n          str(object='') -> str\n          str(bytes_or_buffer[, encoding[, errors]]) -> str\n\n          Create a new string object from the given object. If encoding or\n          errors is specified, then the object must expose a data buffer\n          that will be decoded using the given encoding and error handler.\n          Otherwise, returns the result of object.__str__() (if defined)\n          or repr(object).\n          encoding defaults to sys.getdefaultencoding().\n          errors defaults to 'strict'.\n\n        "
        if len(args) == 0:
            return super(newstr, cls).__new__(cls)
        elif type(args[0]) == newstr and cls == newstr:
            return args[0]
        elif isinstance(args[0], unicode):
            value = args[0]
        elif isinstance(args[0], bytes):
            if 'encoding' in kwargs or len(args) > 1:
                value = args[0].decode(*args[1:], **kwargs)
            else:
                value = args[0].__str__()
        else:
            value = args[0]
        return super(newstr, cls).__new__(cls, value)

    def __repr__(self):
        if False:
            return 10
        '\n        Without the u prefix\n        '
        value = super(newstr, self).__repr__()
        return value[1:]

    def __getitem__(self, y):
        if False:
            return 10
        '\n        Warning: Python <= 2.7.6 has a bug that causes this method never to be called\n        when y is a slice object. Therefore the type of newstr()[:2] is wrong\n        (unicode instead of newstr).\n        '
        return newstr(super(newstr, self).__getitem__(y))

    def __contains__(self, key):
        if False:
            print('Hello World!')
        errmsg = "'in <string>' requires string as left operand, not {0}"
        if type(key) == newstr:
            newkey = key
        elif isinstance(key, unicode) or (isinstance(key, bytes) and (not isnewbytes(key))):
            newkey = newstr(key)
        else:
            raise TypeError(errmsg.format(type(key)))
        return issubset(list(newkey), list(self))

    @no('newbytes')
    def __add__(self, other):
        if False:
            print('Hello World!')
        return newstr(super(newstr, self).__add__(other))

    @no('newbytes')
    def __radd__(self, left):
        if False:
            for i in range(10):
                print('nop')
        ' left + self '
        try:
            return newstr(left) + self
        except:
            return NotImplemented

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        return newstr(super(newstr, self).__mul__(other))

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return newstr(super(newstr, self).__rmul__(other))

    def join(self, iterable):
        if False:
            return 10
        errmsg = 'sequence item {0}: expected unicode string, found bytes'
        for (i, item) in enumerate(iterable):
            if isnewbytes(item):
                raise TypeError(errmsg.format(i))
        if type(self) == newstr:
            return newstr(super(newstr, self).join(iterable))
        else:
            return newstr(super(newstr, newstr(self)).join(iterable))

    @no('newbytes')
    def find(self, sub, *args):
        if False:
            i = 10
            return i + 15
        return super(newstr, self).find(sub, *args)

    @no('newbytes')
    def rfind(self, sub, *args):
        if False:
            i = 10
            return i + 15
        return super(newstr, self).rfind(sub, *args)

    @no('newbytes', (1, 2))
    def replace(self, old, new, *args):
        if False:
            for i in range(10):
                print('nop')
        return newstr(super(newstr, self).replace(old, new, *args))

    def decode(self, *args):
        if False:
            i = 10
            return i + 15
        raise AttributeError('decode method has been disabled in newstr')

    def encode(self, encoding='utf-8', errors='strict'):
        if False:
            while True:
                i = 10
        "\n        Returns bytes\n\n        Encode S using the codec registered for encoding. Default encoding\n        is 'utf-8'. errors may be given to set a different error\n        handling scheme. Default is 'strict' meaning that encoding errors raise\n        a UnicodeEncodeError. Other possible values are 'ignore', 'replace' and\n        'xmlcharrefreplace' as well as any other name registered with\n        codecs.register_error that can handle UnicodeEncodeErrors.\n        "
        from future.types.newbytes import newbytes
        if errors == 'surrogateescape':
            if encoding == 'utf-16':
                raise NotImplementedError('FIXME: surrogateescape handling is not yet implemented properly')
            mybytes = []
            for c in self:
                code = ord(c)
                if 55296 <= code <= 56575:
                    mybytes.append(newbytes([code - 56320]))
                else:
                    mybytes.append(c.encode(encoding=encoding))
            return newbytes(b'').join(mybytes)
        return newbytes(super(newstr, self).encode(encoding, errors))

    @no('newbytes', 1)
    def startswith(self, prefix, *args):
        if False:
            return 10
        if isinstance(prefix, Iterable):
            for thing in prefix:
                if isnewbytes(thing):
                    raise TypeError(self.no_convert_msg.format(type(thing)))
        return super(newstr, self).startswith(prefix, *args)

    @no('newbytes', 1)
    def endswith(self, prefix, *args):
        if False:
            while True:
                i = 10
        if isinstance(prefix, Iterable):
            for thing in prefix:
                if isnewbytes(thing):
                    raise TypeError(self.no_convert_msg.format(type(thing)))
        return super(newstr, self).endswith(prefix, *args)

    @no('newbytes', 1)
    def split(self, sep=None, maxsplit=-1):
        if False:
            while True:
                i = 10
        parts = super(newstr, self).split(sep, maxsplit)
        return [newstr(part) for part in parts]

    @no('newbytes', 1)
    def rsplit(self, sep=None, maxsplit=-1):
        if False:
            return 10
        parts = super(newstr, self).rsplit(sep, maxsplit)
        return [newstr(part) for part in parts]

    @no('newbytes', 1)
    def partition(self, sep):
        if False:
            i = 10
            return i + 15
        parts = super(newstr, self).partition(sep)
        return tuple((newstr(part) for part in parts))

    @no('newbytes', 1)
    def rpartition(self, sep):
        if False:
            i = 10
            return i + 15
        parts = super(newstr, self).rpartition(sep)
        return tuple((newstr(part) for part in parts))

    @no('newbytes', 1)
    def index(self, sub, *args):
        if False:
            print('Hello World!')
        '\n        Like newstr.find() but raise ValueError when the substring is not\n        found.\n        '
        pos = self.find(sub, *args)
        if pos == -1:
            raise ValueError('substring not found')
        return pos

    def splitlines(self, keepends=False):
        if False:
            i = 10
            return i + 15
        '\n        S.splitlines(keepends=False) -> list of strings\n\n        Return a list of the lines in S, breaking at line boundaries.\n        Line breaks are not included in the resulting list unless keepends\n        is given and true.\n        '
        parts = super(newstr, self).splitlines(keepends)
        return [newstr(part) for part in parts]

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, unicode) or (isinstance(other, bytes) and (not isnewbytes(other))):
            return super(newstr, self).__eq__(other)
        else:
            return NotImplemented

    def __hash__(self):
        if False:
            while True:
                i = 10
        if isinstance(self, unicode) or (isinstance(self, bytes) and (not isnewbytes(self))):
            return super(newstr, self).__hash__()
        else:
            raise NotImplementedError()

    def __ne__(self, other):
        if False:
            return 10
        if isinstance(other, unicode) or (isinstance(other, bytes) and (not isnewbytes(other))):
            return super(newstr, self).__ne__(other)
        else:
            return True
    unorderable_err = 'unorderable types: str() and {0}'

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, unicode) or (isinstance(other, bytes) and (not isnewbytes(other))):
            return super(newstr, self).__lt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __le__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, unicode) or (isinstance(other, bytes) and (not isnewbytes(other))):
            return super(newstr, self).__le__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, unicode) or (isinstance(other, bytes) and (not isnewbytes(other))):
            return super(newstr, self).__gt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __ge__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, unicode) or (isinstance(other, bytes) and (not isnewbytes(other))):
            return super(newstr, self).__ge__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __getattribute__(self, name):
        if False:
            i = 10
            return i + 15
        "\n        A trick to cause the ``hasattr`` builtin-fn to return False for\n        the 'decode' method on Py2.\n        "
        if name in ['decode', u'decode']:
            raise AttributeError('decode method has been disabled in newstr')
        return super(newstr, self).__getattribute__(name)

    def __native__(self):
        if False:
            i = 10
            return i + 15
        '\n        A hook for the future.utils.native() function.\n        '
        return unicode(self)

    @staticmethod
    def maketrans(x, y=None, z=None):
        if False:
            return 10
        '\n        Return a translation table usable for str.translate().\n\n        If there is only one argument, it must be a dictionary mapping Unicode\n        ordinals (integers) or characters to Unicode ordinals, strings or None.\n        Character keys will be then converted to ordinals.\n        If there are two arguments, they must be strings of equal length, and\n        in the resulting dictionary, each character in x will be mapped to the\n        character at the same position in y. If there is a third argument, it\n        must be a string, whose characters will be mapped to None in the result.\n        '
        if y is None:
            assert z is None
            if not isinstance(x, dict):
                raise TypeError('if you give only one argument to maketrans it must be a dict')
            result = {}
            for (key, value) in x.items():
                if len(key) > 1:
                    raise ValueError('keys in translate table must be strings or integers')
                result[ord(key)] = value
        else:
            if not isinstance(x, unicode) and isinstance(y, unicode):
                raise TypeError('x and y must be unicode strings')
            if not len(x) == len(y):
                raise ValueError('the first two maketrans arguments must have equal length')
            result = {}
            for (xi, yi) in zip(x, y):
                if len(xi) > 1:
                    raise ValueError('keys in translate table must be strings or integers')
                result[ord(xi)] = ord(yi)
        if z is not None:
            for char in z:
                result[ord(char)] = None
        return result

    def translate(self, table):
        if False:
            print('Hello World!')
        '\n        S.translate(table) -> str\n\n        Return a copy of the string S, where all characters have been mapped\n        through the given translation table, which must be a mapping of\n        Unicode ordinals to Unicode ordinals, strings, or None.\n        Unmapped characters are left untouched. Characters mapped to None\n        are deleted.\n        '
        l = []
        for c in self:
            if ord(c) in table:
                val = table[ord(c)]
                if val is None:
                    continue
                elif isinstance(val, unicode):
                    l.append(val)
                else:
                    l.append(chr(val))
            else:
                l.append(c)
        return ''.join(l)

    def isprintable(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('fixme')

    def isidentifier(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('fixme')

    def format_map(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('fixme')
__all__ = ['newstr']