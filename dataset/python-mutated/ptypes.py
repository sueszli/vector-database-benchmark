"""
This module defines packable types, that is types than can be easily
converted to a binary format as used in MachO headers.
"""
import struct
import sys
from itertools import chain, starmap
try:
    from itertools import imap, izip
except ImportError:
    (izip, imap) = (zip, map)
__all__ = '\nsizeof\nBasePackable\nStructure\npypackable\np_char\np_byte\np_ubyte\np_short\np_ushort\np_int\np_uint\np_long\np_ulong\np_longlong\np_ulonglong\np_int8\np_uint8\np_int16\np_uint16\np_int32\np_uint32\np_int64\np_uint64\np_float\np_double\n'.split()

def sizeof(s):
    if False:
        i = 10
        return i + 15
    '\n    Return the size of an object when packed\n    '
    if hasattr(s, '_size_'):
        return s._size_
    elif isinstance(s, bytes):
        return len(s)
    raise ValueError(s)

class MetaPackable(type):
    """
    Fixed size struct.unpack-able types use from_tuple as their designated
    initializer
    """

    def from_mmap(cls, mm, ptr, **kw):
        if False:
            i = 10
            return i + 15
        return cls.from_str(mm[ptr:ptr + cls._size_], **kw)

    def from_fileobj(cls, f, **kw):
        if False:
            return 10
        return cls.from_str(f.read(cls._size_), **kw)

    def from_str(cls, s, **kw):
        if False:
            return 10
        endian = kw.get('_endian_', cls._endian_)
        return cls.from_tuple(struct.unpack(endian + cls._format_, s), **kw)

    def from_tuple(cls, tpl, **kw):
        if False:
            while True:
                i = 10
        return cls(tpl[0], **kw)

class BasePackable(object):
    _endian_ = '>'

    def to_str(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def to_fileobj(self, f):
        if False:
            while True:
                i = 10
        f.write(self.to_str())

    def to_mmap(self, mm, ptr):
        if False:
            for i in range(10):
                print('nop')
        mm[ptr:ptr + self._size_] = self.to_str()

def _make():
    if False:
        i = 10
        return i + 15

    def to_str(self):
        if False:
            print('Hello World!')
        cls = type(self)
        endian = getattr(self, '_endian_', cls._endian_)
        return struct.pack(endian + cls._format_, self)
    return MetaPackable('Packable', (BasePackable,), {'to_str': to_str})
Packable = _make()
del _make

def pypackable(name, pytype, format):
    if False:
        print('Hello World!')
    '\n    Create a "mix-in" class with a python type and a\n    Packable with the given struct format\n    '
    (size, items) = _formatinfo(format)

    def __new__(cls, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        if '_endian_' in kwds:
            _endian_ = kwds.pop('_endian_')
        else:
            _endian_ = cls._endian_
        result = pytype.__new__(cls, *args, **kwds)
        result._endian_ = _endian_
        return result
    return type(Packable)(name, (pytype, Packable), {'_format_': format, '_size_': size, '_items_': items, '__new__': __new__})

def _formatinfo(format):
    if False:
        i = 10
        return i + 15
    '\n    Calculate the size and number of items in a struct format.\n    '
    size = struct.calcsize(format)
    return (size, len(struct.unpack(format, b'\x00' * size)))

class MetaStructure(MetaPackable):
    """
    The metaclass of Structure objects that does all the magic.

    Since we can assume that all Structures have a fixed size,
    we can do a bunch of calculations up front and pack or
    unpack the whole thing in one struct call.
    """

    def __new__(cls, clsname, bases, dct):
        if False:
            while True:
                i = 10
        fields = dct['_fields_']
        names = []
        types = []
        structmarks = []
        format = ''
        items = 0
        size = 0

        def struct_property(name, typ):
            if False:
                print('Hello World!')

            def _get(self):
                if False:
                    print('Hello World!')
                return self._objects_[name]

            def _set(self, obj):
                if False:
                    return 10
                if type(obj) is not typ:
                    obj = typ(obj)
                self._objects_[name] = obj
            return property(_get, _set, typ.__name__)
        for (name, typ) in fields:
            dct[name] = struct_property(name, typ)
            names.append(name)
            types.append(typ)
            format += typ._format_
            size += typ._size_
            if typ._items_ > 1:
                structmarks.append((items, typ._items_, typ))
            items += typ._items_
        dct['_structmarks_'] = structmarks
        dct['_names_'] = names
        dct['_types_'] = types
        dct['_size_'] = size
        dct['_items_'] = items
        dct['_format_'] = format
        return super(MetaStructure, cls).__new__(cls, clsname, bases, dct)

    def from_tuple(cls, tpl, **kw):
        if False:
            i = 10
            return i + 15
        values = []
        current = 0
        for (begin, length, typ) in cls._structmarks_:
            if begin > current:
                values.extend(tpl[current:begin])
            current = begin + length
            values.append(typ.from_tuple(tpl[begin:current], **kw))
        values.extend(tpl[current:])
        return cls(*values, **kw)

def _make():
    if False:
        for i in range(10):
            print('nop')
    class_dict = {}
    class_dict['_fields_'] = ()

    def as_method(function):
        if False:
            while True:
                i = 10
        class_dict[function.__name__] = function

    @as_method
    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if len(args) == 1 and (not kwargs) and (type(args[0]) is type(self)):
            kwargs = args[0]._objects_
            args = ()
        self._objects_ = {}
        iargs = chain(izip(self._names_, args), kwargs.items())
        for (key, value) in iargs:
            if key not in self._names_ and key != '_endian_':
                raise TypeError
            setattr(self, key, value)
        for (key, typ) in izip(self._names_, self._types_):
            if key not in self._objects_:
                self._objects_[key] = typ()

    @as_method
    def _get_packables(self):
        if False:
            print('Hello World!')
        for obj in imap(self._objects_.__getitem__, self._names_):
            if hasattr(obj, '_get_packables'):
                for obj in obj._get_packables():
                    yield obj
            else:
                yield obj

    @as_method
    def to_str(self):
        if False:
            i = 10
            return i + 15
        return struct.pack(self._endian_ + self._format_, *self._get_packables())

    @as_method
    def __cmp__(self, other):
        if False:
            return 10
        if type(other) is not type(self):
            raise TypeError('Cannot compare objects of type %r to objects of type %r' % (type(other), type(self)))
        if sys.version_info[0] == 2:
            _cmp = cmp
        else:

            def _cmp(a, b):
                if False:
                    i = 10
                    return i + 15
                if a < b:
                    return -1
                elif a > b:
                    return 1
                elif a == b:
                    return 0
                else:
                    raise TypeError()
        for cmpval in starmap(_cmp, izip(self._get_packables(), other._get_packables())):
            if cmpval != 0:
                return cmpval
        return 0

    @as_method
    def __eq__(self, other):
        if False:
            return 10
        r = self.__cmp__(other)
        return r == 0

    @as_method
    def __ne__(self, other):
        if False:
            return 10
        r = self.__cmp__(other)
        return r != 0

    @as_method
    def __lt__(self, other):
        if False:
            while True:
                i = 10
        r = self.__cmp__(other)
        return r < 0

    @as_method
    def __le__(self, other):
        if False:
            return 10
        r = self.__cmp__(other)
        return r <= 0

    @as_method
    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        r = self.__cmp__(other)
        return r > 0

    @as_method
    def __ge__(self, other):
        if False:
            while True:
                i = 10
        r = self.__cmp__(other)
        return r >= 0

    @as_method
    def __repr__(self):
        if False:
            print('Hello World!')
        result = []
        result.append('<')
        result.append(type(self).__name__)
        for nm in self._names_:
            result.append(' %s=%r' % (nm, getattr(self, nm)))
        result.append('>')
        return ''.join(result)
    return MetaStructure('Structure', (BasePackable,), class_dict)
Structure = _make()
del _make
try:
    long
except NameError:
    long = int
p_char = pypackable('p_char', bytes, 'c')
p_int8 = pypackable('p_int8', int, 'b')
p_uint8 = pypackable('p_uint8', int, 'B')
p_int16 = pypackable('p_int16', int, 'h')
p_uint16 = pypackable('p_uint16', int, 'H')
p_int32 = pypackable('p_int32', int, 'i')
p_uint32 = pypackable('p_uint32', long, 'I')
p_int64 = pypackable('p_int64', long, 'q')
p_uint64 = pypackable('p_uint64', long, 'Q')
p_float = pypackable('p_float', float, 'f')
p_double = pypackable('p_double', float, 'd')
p_byte = p_int8
p_ubyte = p_uint8
p_short = p_int16
p_ushort = p_uint16
p_int = p_long = p_int32
p_uint = p_ulong = p_uint32
p_longlong = p_int64
p_ulonglong = p_uint64