"""
u-msgpack-python v2.7.1 - v at sergeev.io
https://github.com/vsergeev/u-msgpack-python

u-msgpack-python is a lightweight MessagePack serializer and deserializer
module, compatible with both Python 2 and 3, as well CPython and PyPy
implementations of Python. u-msgpack-python is fully compliant with the
latest MessagePack specification.com/msgpack/msgpack/blob/master/spec.md). In
particular, it supports the new binary, UTF-8 string, and application ext
types.

License: MIT
"""
import struct
import collections
import datetime
import sys
import io
if sys.version_info[0:2] >= (3, 3):
    from collections.abc import Hashable
else:
    from collections import Hashable
__version__ = '2.7.1'
'Module version string'
version = (2, 7, 1)
'Module version tuple'

class Ext(object):
    """
    The Ext class facilitates creating a serializable extension object to store
    an application-defined type and data byte array.
    """

    def __init__(self, type, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new Ext object.\n\n        Args:\n            type: application-defined type integer\n            data: application-defined data byte array\n\n        TypeError:\n            Type is not an integer.\n        ValueError:\n            Type is out of range of -128 to 127.\n        TypeError::\n            Data is not type \'bytes\' (Python 3) or not type \'str\' (Python 2).\n\n        Example:\n        >>> foo = umsgpack.Ext(5, b"\x01\x02\x03")\n        >>> umsgpack.packb({u"special stuff": foo, u"awesome": True})\n        \'\x82§awesomeÃ\xadspecial stuffÇ\x03\x05\x01\x02\x03\'\n        >>> bar = umsgpack.unpackb(_)\n        >>> print(bar["special stuff"])\n        Ext Object (Type: 5, Data: 01 02 03)\n        >>>\n        '
        if not isinstance(type, int):
            raise TypeError('ext type is not type integer')
        elif not -2 ** 7 <= type <= 2 ** 7 - 1:
            raise ValueError('ext type value {:d} is out of range (-128 to 127)'.format(type))
        elif sys.version_info[0] == 3 and (not isinstance(data, bytes)):
            raise TypeError("ext data is not type 'bytes'")
        elif sys.version_info[0] == 2 and (not isinstance(data, str)):
            raise TypeError("ext data is not type 'str'")
        self.type = type
        self.data = data

    def __eq__(self, other):
        if False:
            return 10
        '\n        Compare this Ext object with another for equality.\n        '
        return isinstance(other, self.__class__) and self.type == other.type and (self.data == other.data)

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        '\n        Compare this Ext object with another for inequality.\n        '
        return not self.__eq__(other)

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        String representation of this Ext object.\n        '
        s = 'Ext Object (Type: {:d}, Data: '.format(self.type)
        s += ' '.join(['0x{:02}'.format(ord(self.data[i:i + 1])) for i in xrange(min(len(self.data), 8))])
        if len(self.data) > 8:
            s += ' ...'
        s += ')'
        return s

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Provide a hash of this Ext object.\n        '
        return hash((self.type, self.data))

class InvalidString(bytes):
    """Subclass of bytes to hold invalid UTF-8 strings."""
_ext_class_to_type = {}
_ext_type_to_class = {}

def ext_serializable(ext_type):
    if False:
        return 10
    '\n    Return a decorator to register a class for automatic packing and unpacking\n    with the specified Ext type code. The application class should implement a\n    `packb()` method that returns serialized bytes, and an `unpackb()` class\n    method or static method that accepts serialized bytes and returns an\n    instance of the application class.\n\n    Args:\n        ext_type: application-defined Ext type code\n\n    Raises:\n        TypeError:\n            Ext type is not an integer.\n        ValueError:\n            Ext type is out of range of -128 to 127.\n        ValueError:\n            Ext type or class already registered.\n    '

    def wrapper(cls):
        if False:
            print('Hello World!')
        if not isinstance(ext_type, int):
            raise TypeError('Ext type is not type integer')
        elif not -2 ** 7 <= ext_type <= 2 ** 7 - 1:
            raise ValueError('Ext type value {:d} is out of range of -128 to 127'.format(ext_type))
        elif ext_type in _ext_type_to_class:
            raise ValueError('Ext type {:d} already registered with class {:s}'.format(ext_type, repr(_ext_type_to_class[ext_type])))
        elif cls in _ext_class_to_type:
            raise ValueError('Class {:s} already registered with Ext type {:d}'.format(repr(cls), ext_type))
        _ext_type_to_class[ext_type] = cls
        _ext_class_to_type[cls] = ext_type
        return cls
    return wrapper

class PackException(Exception):
    """Base class for exceptions encountered during packing."""

class UnpackException(Exception):
    """Base class for exceptions encountered during unpacking."""

class UnsupportedTypeException(PackException):
    """Object type not supported for packing."""

class InsufficientDataException(UnpackException):
    """Insufficient data to unpack the serialized object."""

class InvalidStringException(UnpackException):
    """Invalid UTF-8 string encountered during unpacking."""

class UnsupportedTimestampException(UnpackException):
    """Unsupported timestamp format encountered during unpacking."""

class ReservedCodeException(UnpackException):
    """Reserved code encountered during unpacking."""

class UnhashableKeyException(UnpackException):
    """
    Unhashable key encountered during map unpacking.
    The serialized map cannot be deserialized into a Python dictionary.
    """

class DuplicateKeyException(UnpackException):
    """Duplicate key encountered during map unpacking."""
KeyNotPrimitiveException = UnhashableKeyException
KeyDuplicateException = DuplicateKeyException
pack = None
packb = None
unpack = None
unpackb = None
dump = None
dumps = None
load = None
loads = None
compatibility = False
'\nCompatibility mode boolean.\n\nWhen compatibility mode is enabled, u-msgpack-python will serialize both\nunicode strings and bytes into the old "raw" msgpack type, and deserialize the\n"raw" msgpack type into bytes. This provides backwards compatibility with the\nold MessagePack specification.\n\nExample:\n>>> umsgpack.compatibility = True\n>>>\n>>> umsgpack.packb([u"some string", b"some bytes"])\nb\'\x92«some stringªsome bytes\'\n>>> umsgpack.unpackb(_)\n[b\'some string\', b\'some bytes\']\n>>>\n'

def _pack_integer(obj, fp, options):
    if False:
        while True:
            i = 10
    if obj < 0:
        if obj >= -32:
            fp.write(struct.pack('b', obj))
        elif obj >= -2 ** (8 - 1):
            fp.write(b'\xd0' + struct.pack('b', obj))
        elif obj >= -2 ** (16 - 1):
            fp.write(b'\xd1' + struct.pack('>h', obj))
        elif obj >= -2 ** (32 - 1):
            fp.write(b'\xd2' + struct.pack('>i', obj))
        elif obj >= -2 ** (64 - 1):
            fp.write(b'\xd3' + struct.pack('>q', obj))
        else:
            raise UnsupportedTypeException('huge signed int')
    elif obj < 128:
        fp.write(struct.pack('B', obj))
    elif obj < 2 ** 8:
        fp.write(b'\xcc' + struct.pack('B', obj))
    elif obj < 2 ** 16:
        fp.write(b'\xcd' + struct.pack('>H', obj))
    elif obj < 2 ** 32:
        fp.write(b'\xce' + struct.pack('>I', obj))
    elif obj < 2 ** 64:
        fp.write(b'\xcf' + struct.pack('>Q', obj))
    else:
        raise UnsupportedTypeException('huge unsigned int')

def _pack_nil(obj, fp, options):
    if False:
        print('Hello World!')
    fp.write(b'\xc0')

def _pack_boolean(obj, fp, options):
    if False:
        for i in range(10):
            print('nop')
    fp.write(b'\xc3' if obj else b'\xc2')

def _pack_float(obj, fp, options):
    if False:
        print('Hello World!')
    float_precision = options.get('force_float_precision', _float_precision)
    if float_precision == 'double':
        fp.write(b'\xcb' + struct.pack('>d', obj))
    elif float_precision == 'single':
        fp.write(b'\xca' + struct.pack('>f', obj))
    else:
        raise ValueError('invalid float precision')

def _pack_string(obj, fp, options):
    if False:
        return 10
    obj = obj.encode('utf-8')
    obj_len = len(obj)
    if obj_len < 32:
        fp.write(struct.pack('B', 160 | obj_len) + obj)
    elif obj_len < 2 ** 8:
        fp.write(b'\xd9' + struct.pack('B', obj_len) + obj)
    elif obj_len < 2 ** 16:
        fp.write(b'\xda' + struct.pack('>H', obj_len) + obj)
    elif obj_len < 2 ** 32:
        fp.write(b'\xdb' + struct.pack('>I', obj_len) + obj)
    else:
        raise UnsupportedTypeException('huge string')

def _pack_binary(obj, fp, options):
    if False:
        while True:
            i = 10
    obj_len = len(obj)
    if obj_len < 2 ** 8:
        fp.write(b'\xc4' + struct.pack('B', obj_len) + obj)
    elif obj_len < 2 ** 16:
        fp.write(b'\xc5' + struct.pack('>H', obj_len) + obj)
    elif obj_len < 2 ** 32:
        fp.write(b'\xc6' + struct.pack('>I', obj_len) + obj)
    else:
        raise UnsupportedTypeException('huge binary string')

def _pack_oldspec_raw(obj, fp, options):
    if False:
        print('Hello World!')
    obj_len = len(obj)
    if obj_len < 32:
        fp.write(struct.pack('B', 160 | obj_len) + obj)
    elif obj_len < 2 ** 16:
        fp.write(b'\xda' + struct.pack('>H', obj_len) + obj)
    elif obj_len < 2 ** 32:
        fp.write(b'\xdb' + struct.pack('>I', obj_len) + obj)
    else:
        raise UnsupportedTypeException('huge raw string')

def _pack_ext(obj, fp, options):
    if False:
        i = 10
        return i + 15
    obj_len = len(obj.data)
    if obj_len == 1:
        fp.write(b'\xd4' + struct.pack('B', obj.type & 255) + obj.data)
    elif obj_len == 2:
        fp.write(b'\xd5' + struct.pack('B', obj.type & 255) + obj.data)
    elif obj_len == 4:
        fp.write(b'\xd6' + struct.pack('B', obj.type & 255) + obj.data)
    elif obj_len == 8:
        fp.write(b'\xd7' + struct.pack('B', obj.type & 255) + obj.data)
    elif obj_len == 16:
        fp.write(b'\xd8' + struct.pack('B', obj.type & 255) + obj.data)
    elif obj_len < 2 ** 8:
        fp.write(b'\xc7' + struct.pack('BB', obj_len, obj.type & 255) + obj.data)
    elif obj_len < 2 ** 16:
        fp.write(b'\xc8' + struct.pack('>HB', obj_len, obj.type & 255) + obj.data)
    elif obj_len < 2 ** 32:
        fp.write(b'\xc9' + struct.pack('>IB', obj_len, obj.type & 255) + obj.data)
    else:
        raise UnsupportedTypeException('huge ext data')

def _pack_ext_timestamp(obj, fp, options):
    if False:
        return 10
    if not obj.tzinfo:
        delta = obj.replace(tzinfo=_utc_tzinfo) - _epoch
    else:
        delta = obj - _epoch
    seconds = delta.seconds + delta.days * 86400
    microseconds = delta.microseconds
    if microseconds == 0 and 0 <= seconds <= 2 ** 32 - 1:
        fp.write(b'\xd6\xff' + struct.pack('>I', seconds))
    elif 0 <= seconds <= 2 ** 34 - 1:
        value = microseconds * 1000 << 34 | seconds
        fp.write(b'\xd7\xff' + struct.pack('>Q', value))
    elif -2 ** 63 <= abs(seconds) <= 2 ** 63 - 1:
        fp.write(b'\xc7\x0c\xff' + struct.pack('>Iq', microseconds * 1000, seconds))
    else:
        raise UnsupportedTypeException('huge timestamp')

def _pack_array(obj, fp, options):
    if False:
        for i in range(10):
            print('nop')
    obj_len = len(obj)
    if obj_len < 16:
        fp.write(struct.pack('B', 144 | obj_len))
    elif obj_len < 2 ** 16:
        fp.write(b'\xdc' + struct.pack('>H', obj_len))
    elif obj_len < 2 ** 32:
        fp.write(b'\xdd' + struct.pack('>I', obj_len))
    else:
        raise UnsupportedTypeException('huge array')
    for e in obj:
        pack(e, fp, **options)

def _pack_map(obj, fp, options):
    if False:
        i = 10
        return i + 15
    obj_len = len(obj)
    if obj_len < 16:
        fp.write(struct.pack('B', 128 | obj_len))
    elif obj_len < 2 ** 16:
        fp.write(b'\xde' + struct.pack('>H', obj_len))
    elif obj_len < 2 ** 32:
        fp.write(b'\xdf' + struct.pack('>I', obj_len))
    else:
        raise UnsupportedTypeException('huge array')
    for (k, v) in obj.items():
        pack(k, fp, **options)
        pack(v, fp, **options)

def _pack2(obj, fp, **options):
    if False:
        print('Hello World!')
    '\n    Serialize a Python object into MessagePack bytes.\n\n    Args:\n        obj: a Python object\n        fp: a .write()-supporting file-like object\n\n    Kwargs:\n        ext_handlers (dict): dictionary of Ext handlers, mapping a custom type\n                             to a callable that packs an instance of the type\n                             into an Ext object\n        force_float_precision (str): "single" to force packing floats as\n                                     IEEE-754 single-precision floats,\n                                     "double" to force packing floats as\n                                     IEEE-754 double-precision floats.\n\n    Returns:\n        None.\n\n    Raises:\n        UnsupportedType(PackException):\n            Object type not supported for packing.\n\n    Example:\n    >>> f = open(\'test.bin\', \'wb\')\n    >>> umsgpack.pack({u"compact": True, u"schema": 0}, f)\n    >>>\n    '
    global compatibility
    ext_handlers = options.get('ext_handlers')
    if obj is None:
        _pack_nil(obj, fp, options)
    elif ext_handlers and obj.__class__ in ext_handlers:
        _pack_ext(ext_handlers[obj.__class__](obj), fp, options)
    elif obj.__class__ in _ext_class_to_type:
        try:
            _pack_ext(Ext(_ext_class_to_type[obj.__class__], obj.packb()), fp, options)
        except AttributeError:
            raise NotImplementedError('Ext serializable class {:s} is missing implementation of packb()'.format(repr(obj.__class__)))
    elif isinstance(obj, bool):
        _pack_boolean(obj, fp, options)
    elif isinstance(obj, (int, long)):
        _pack_integer(obj, fp, options)
    elif isinstance(obj, float):
        _pack_float(obj, fp, options)
    elif compatibility and isinstance(obj, unicode):
        _pack_oldspec_raw(bytes(obj), fp, options)
    elif compatibility and isinstance(obj, bytes):
        _pack_oldspec_raw(obj, fp, options)
    elif isinstance(obj, unicode):
        _pack_string(obj, fp, options)
    elif isinstance(obj, str):
        _pack_binary(obj, fp, options)
    elif isinstance(obj, (list, tuple)):
        _pack_array(obj, fp, options)
    elif isinstance(obj, dict):
        _pack_map(obj, fp, options)
    elif isinstance(obj, datetime.datetime):
        _pack_ext_timestamp(obj, fp, options)
    elif isinstance(obj, Ext):
        _pack_ext(obj, fp, options)
    elif ext_handlers:
        t = next((t for t in ext_handlers.keys() if isinstance(obj, t)), None)
        if t:
            _pack_ext(ext_handlers[t](obj), fp, options)
        else:
            raise UnsupportedTypeException('unsupported type: {:s}'.format(str(type(obj))))
    elif _ext_class_to_type:
        t = next((t for t in _ext_class_to_type if isinstance(obj, t)), None)
        if t:
            try:
                _pack_ext(Ext(_ext_class_to_type[t], obj.packb()), fp, options)
            except AttributeError:
                raise NotImplementedError('Ext serializable class {:s} is missing implementation of packb()'.format(repr(t)))
        else:
            raise UnsupportedTypeException('unsupported type: {:s}'.format(str(type(obj))))
    else:
        raise UnsupportedTypeException('unsupported type: {:s}'.format(str(type(obj))))

def _pack3(obj, fp, **options):
    if False:
        for i in range(10):
            print('nop')
    '\n    Serialize a Python object into MessagePack bytes.\n\n    Args:\n        obj: a Python object\n        fp: a .write()-supporting file-like object\n\n    Kwargs:\n        ext_handlers (dict): dictionary of Ext handlers, mapping a custom type\n                             to a callable that packs an instance of the type\n                             into an Ext object\n        force_float_precision (str): "single" to force packing floats as\n                                     IEEE-754 single-precision floats,\n                                     "double" to force packing floats as\n                                     IEEE-754 double-precision floats.\n\n    Returns:\n        None.\n\n    Raises:\n        UnsupportedType(PackException):\n            Object type not supported for packing.\n\n    Example:\n    >>> f = open(\'test.bin\', \'wb\')\n    >>> umsgpack.pack({u"compact": True, u"schema": 0}, f)\n    >>>\n    '
    global compatibility
    ext_handlers = options.get('ext_handlers')
    if obj is None:
        _pack_nil(obj, fp, options)
    elif ext_handlers and obj.__class__ in ext_handlers:
        _pack_ext(ext_handlers[obj.__class__](obj), fp, options)
    elif obj.__class__ in _ext_class_to_type:
        try:
            _pack_ext(Ext(_ext_class_to_type[obj.__class__], obj.packb()), fp, options)
        except AttributeError:
            raise NotImplementedError('Ext serializable class {:s} is missing implementation of packb()'.format(repr(obj.__class__)))
    elif isinstance(obj, bool):
        _pack_boolean(obj, fp, options)
    elif isinstance(obj, int):
        _pack_integer(obj, fp, options)
    elif isinstance(obj, float):
        _pack_float(obj, fp, options)
    elif compatibility and isinstance(obj, str):
        _pack_oldspec_raw(obj.encode('utf-8'), fp, options)
    elif compatibility and isinstance(obj, bytes):
        _pack_oldspec_raw(obj, fp, options)
    elif isinstance(obj, str):
        _pack_string(obj, fp, options)
    elif isinstance(obj, bytes):
        _pack_binary(obj, fp, options)
    elif isinstance(obj, (list, tuple)):
        _pack_array(obj, fp, options)
    elif isinstance(obj, dict):
        _pack_map(obj, fp, options)
    elif isinstance(obj, datetime.datetime):
        _pack_ext_timestamp(obj, fp, options)
    elif isinstance(obj, Ext):
        _pack_ext(obj, fp, options)
    elif ext_handlers:
        t = next((t for t in ext_handlers.keys() if isinstance(obj, t)), None)
        if t:
            _pack_ext(ext_handlers[t](obj), fp, options)
        else:
            raise UnsupportedTypeException('unsupported type: {:s}'.format(str(type(obj))))
    elif _ext_class_to_type:
        t = next((t for t in _ext_class_to_type if isinstance(obj, t)), None)
        if t:
            try:
                _pack_ext(Ext(_ext_class_to_type[t], obj.packb()), fp, options)
            except AttributeError:
                raise NotImplementedError('Ext serializable class {:s} is missing implementation of packb()'.format(repr(t)))
        else:
            raise UnsupportedTypeException('unsupported type: {:s}'.format(str(type(obj))))
    else:
        raise UnsupportedTypeException('unsupported type: {:s}'.format(str(type(obj))))

def _packb2(obj, **options):
    if False:
        for i in range(10):
            print('nop')
    '\n    Serialize a Python object into MessagePack bytes.\n\n    Args:\n        obj: a Python object\n\n    Kwargs:\n        ext_handlers (dict): dictionary of Ext handlers, mapping a custom type\n                             to a callable that packs an instance of the type\n                             into an Ext object\n        force_float_precision (str): "single" to force packing floats as\n                                     IEEE-754 single-precision floats,\n                                     "double" to force packing floats as\n                                     IEEE-754 double-precision floats.\n\n    Returns:\n        A \'str\' containing serialized MessagePack bytes.\n\n    Raises:\n        UnsupportedType(PackException):\n            Object type not supported for packing.\n\n    Example:\n    >>> umsgpack.packb({u"compact": True, u"schema": 0})\n    \'\x82§compactÃ¦schema\x00\'\n    >>>\n    '
    fp = io.BytesIO()
    _pack2(obj, fp, **options)
    return fp.getvalue()

def _packb3(obj, **options):
    if False:
        for i in range(10):
            print('nop')
    '\n    Serialize a Python object into MessagePack bytes.\n\n    Args:\n        obj: a Python object\n\n    Kwargs:\n        ext_handlers (dict): dictionary of Ext handlers, mapping a custom type\n                             to a callable that packs an instance of the type\n                             into an Ext object\n        force_float_precision (str): "single" to force packing floats as\n                                     IEEE-754 single-precision floats,\n                                     "double" to force packing floats as\n                                     IEEE-754 double-precision floats.\n\n    Returns:\n        A \'bytes\' containing serialized MessagePack bytes.\n\n    Raises:\n        UnsupportedType(PackException):\n            Object type not supported for packing.\n\n    Example:\n    >>> umsgpack.packb({u"compact": True, u"schema": 0})\n    b\'\x82§compactÃ¦schema\x00\'\n    >>>\n    '
    fp = io.BytesIO()
    _pack3(obj, fp, **options)
    return fp.getvalue()

def _read_except(fp, n):
    if False:
        print('Hello World!')
    if n == 0:
        return b''
    data = fp.read(n)
    if len(data) == 0:
        raise InsufficientDataException()
    while len(data) < n:
        chunk = fp.read(n - len(data))
        if len(chunk) == 0:
            raise InsufficientDataException()
        data += chunk
    return data

def _unpack_integer(code, fp, options):
    if False:
        return 10
    if ord(code) & 224 == 224:
        return struct.unpack('b', code)[0]
    elif code == b'\xd0':
        return struct.unpack('b', _read_except(fp, 1))[0]
    elif code == b'\xd1':
        return struct.unpack('>h', _read_except(fp, 2))[0]
    elif code == b'\xd2':
        return struct.unpack('>i', _read_except(fp, 4))[0]
    elif code == b'\xd3':
        return struct.unpack('>q', _read_except(fp, 8))[0]
    elif ord(code) & 128 == 0:
        return struct.unpack('B', code)[0]
    elif code == b'\xcc':
        return struct.unpack('B', _read_except(fp, 1))[0]
    elif code == b'\xcd':
        return struct.unpack('>H', _read_except(fp, 2))[0]
    elif code == b'\xce':
        return struct.unpack('>I', _read_except(fp, 4))[0]
    elif code == b'\xcf':
        return struct.unpack('>Q', _read_except(fp, 8))[0]
    raise Exception('logic error, not int: 0x{:02x}'.format(ord(code)))

def _unpack_reserved(code, fp, options):
    if False:
        return 10
    if code == b'\xc1':
        raise ReservedCodeException('encountered reserved code: 0x{:02x}'.format(ord(code)))
    raise Exception('logic error, not reserved code: 0x{:02x}'.format(ord(code)))

def _unpack_nil(code, fp, options):
    if False:
        i = 10
        return i + 15
    if code == b'\xc0':
        return None
    raise Exception('logic error, not nil: 0x{:02x}'.format(ord(code)))

def _unpack_boolean(code, fp, options):
    if False:
        while True:
            i = 10
    if code == b'\xc2':
        return False
    elif code == b'\xc3':
        return True
    raise Exception('logic error, not boolean: 0x{:02x}'.format(ord(code)))

def _unpack_float(code, fp, options):
    if False:
        return 10
    if code == b'\xca':
        return struct.unpack('>f', _read_except(fp, 4))[0]
    elif code == b'\xcb':
        return struct.unpack('>d', _read_except(fp, 8))[0]
    raise Exception('logic error, not float: 0x{:02x}'.format(ord(code)))

def _unpack_string(code, fp, options):
    if False:
        i = 10
        return i + 15
    if ord(code) & 224 == 160:
        length = ord(code) & ~224
    elif code == b'\xd9':
        length = struct.unpack('B', _read_except(fp, 1))[0]
    elif code == b'\xda':
        length = struct.unpack('>H', _read_except(fp, 2))[0]
    elif code == b'\xdb':
        length = struct.unpack('>I', _read_except(fp, 4))[0]
    else:
        raise Exception('logic error, not string: 0x{:02x}'.format(ord(code)))
    global compatibility
    if compatibility:
        return _read_except(fp, length)
    data = _read_except(fp, length)
    try:
        return bytes.decode(data, 'utf-8')
    except UnicodeDecodeError:
        if options.get('allow_invalid_utf8'):
            return InvalidString(data)
        raise InvalidStringException('unpacked string is invalid utf-8')

def _unpack_binary(code, fp, options):
    if False:
        print('Hello World!')
    if code == b'\xc4':
        length = struct.unpack('B', _read_except(fp, 1))[0]
    elif code == b'\xc5':
        length = struct.unpack('>H', _read_except(fp, 2))[0]
    elif code == b'\xc6':
        length = struct.unpack('>I', _read_except(fp, 4))[0]
    else:
        raise Exception('logic error, not binary: 0x{:02x}'.format(ord(code)))
    return _read_except(fp, length)

def _unpack_ext(code, fp, options):
    if False:
        for i in range(10):
            print('nop')
    if code == b'\xd4':
        length = 1
    elif code == b'\xd5':
        length = 2
    elif code == b'\xd6':
        length = 4
    elif code == b'\xd7':
        length = 8
    elif code == b'\xd8':
        length = 16
    elif code == b'\xc7':
        length = struct.unpack('B', _read_except(fp, 1))[0]
    elif code == b'\xc8':
        length = struct.unpack('>H', _read_except(fp, 2))[0]
    elif code == b'\xc9':
        length = struct.unpack('>I', _read_except(fp, 4))[0]
    else:
        raise Exception('logic error, not ext: 0x{:02x}'.format(ord(code)))
    ext_type = struct.unpack('b', _read_except(fp, 1))[0]
    ext_data = _read_except(fp, length)
    ext_handlers = options.get('ext_handlers')
    if ext_handlers and ext_type in ext_handlers:
        return ext_handlers[ext_type](Ext(ext_type, ext_data))
    if ext_type in _ext_type_to_class:
        try:
            return _ext_type_to_class[ext_type].unpackb(ext_data)
        except AttributeError:
            raise NotImplementedError('Ext serializable class {:s} is missing implementation of unpackb()'.format(repr(_ext_type_to_class[ext_type])))
    if ext_type == -1:
        return _unpack_ext_timestamp(ext_data, options)
    return Ext(ext_type, ext_data)

def _unpack_ext_timestamp(ext_data, options):
    if False:
        while True:
            i = 10
    obj_len = len(ext_data)
    if obj_len == 4:
        seconds = struct.unpack('>I', ext_data)[0]
        microseconds = 0
    elif obj_len == 8:
        value = struct.unpack('>Q', ext_data)[0]
        seconds = value & 17179869183
        microseconds = (value >> 34) // 1000
    elif obj_len == 12:
        seconds = struct.unpack('>q', ext_data[4:12])[0]
        microseconds = struct.unpack('>I', ext_data[0:4])[0] // 1000
    else:
        raise UnsupportedTimestampException('unsupported timestamp with data length {:d}'.format(len(ext_data)))
    return _epoch + datetime.timedelta(seconds=seconds, microseconds=microseconds)

def _unpack_array(code, fp, options):
    if False:
        while True:
            i = 10
    if ord(code) & 240 == 144:
        length = ord(code) & ~240
    elif code == b'\xdc':
        length = struct.unpack('>H', _read_except(fp, 2))[0]
    elif code == b'\xdd':
        length = struct.unpack('>I', _read_except(fp, 4))[0]
    else:
        raise Exception('logic error, not array: 0x{:02x}'.format(ord(code)))
    if options.get('use_tuple'):
        return tuple((_unpack(fp, options) for i in xrange(length)))
    return [_unpack(fp, options) for i in xrange(length)]

def _deep_list_to_tuple(obj):
    if False:
        i = 10
        return i + 15
    if isinstance(obj, list):
        return tuple([_deep_list_to_tuple(e) for e in obj])
    return obj

def _unpack_map(code, fp, options):
    if False:
        while True:
            i = 10
    if ord(code) & 240 == 128:
        length = ord(code) & ~240
    elif code == b'\xde':
        length = struct.unpack('>H', _read_except(fp, 2))[0]
    elif code == b'\xdf':
        length = struct.unpack('>I', _read_except(fp, 4))[0]
    else:
        raise Exception('logic error, not map: 0x{:02x}'.format(ord(code)))
    d = {} if not options.get('use_ordered_dict') else collections.OrderedDict()
    for _ in xrange(length):
        k = _unpack(fp, options)
        if isinstance(k, list):
            k = _deep_list_to_tuple(k)
        elif not isinstance(k, Hashable):
            raise UnhashableKeyException('encountered unhashable key: "{:s}" ({:s})'.format(str(k), str(type(k))))
        elif k in d:
            raise DuplicateKeyException('encountered duplicate key: "{:s}" ({:s})'.format(str(k), str(type(k))))
        v = _unpack(fp, options)
        try:
            d[k] = v
        except TypeError:
            raise UnhashableKeyException('encountered unhashable key: "{:s}"'.format(str(k)))
    return d

def _unpack(fp, options):
    if False:
        while True:
            i = 10
    code = _read_except(fp, 1)
    return _unpack_dispatch_table[code](code, fp, options)

def _unpack2(fp, **options):
    if False:
        i = 10
        return i + 15
    "\n    Deserialize MessagePack bytes into a Python object.\n\n    Args:\n        fp: a .read()-supporting file-like object\n\n    Kwargs:\n        ext_handlers (dict): dictionary of Ext handlers, mapping integer Ext\n                             type to a callable that unpacks an instance of\n                             Ext into an object\n        use_ordered_dict (bool): unpack maps into OrderedDict, instead of\n                                 unordered dict (default False)\n        use_tuple (bool): unpacks arrays into tuples, instead of lists (default\n                          False)\n        allow_invalid_utf8 (bool): unpack invalid strings into instances of\n                                   InvalidString, for access to the bytes\n                                   (default False)\n\n    Returns:\n        A Python object.\n\n    Raises:\n        InsufficientDataException(UnpackException):\n            Insufficient data to unpack the serialized object.\n        InvalidStringException(UnpackException):\n            Invalid UTF-8 string encountered during unpacking.\n        UnsupportedTimestampException(UnpackException):\n            Unsupported timestamp format encountered during unpacking.\n        ReservedCodeException(UnpackException):\n            Reserved code encountered during unpacking.\n        UnhashableKeyException(UnpackException):\n            Unhashable key encountered during map unpacking.\n            The serialized map cannot be deserialized into a Python dictionary.\n        DuplicateKeyException(UnpackException):\n            Duplicate key encountered during map unpacking.\n\n    Example:\n    >>> f = open('test.bin', 'rb')\n    >>> umsgpack.unpackb(f)\n    {u'compact': True, u'schema': 0}\n    >>>\n    "
    return _unpack(fp, options)

def _unpack3(fp, **options):
    if False:
        print('Hello World!')
    "\n    Deserialize MessagePack bytes into a Python object.\n\n    Args:\n        fp: a .read()-supporting file-like object\n\n    Kwargs:\n        ext_handlers (dict): dictionary of Ext handlers, mapping integer Ext\n                             type to a callable that unpacks an instance of\n                             Ext into an object\n        use_ordered_dict (bool): unpack maps into OrderedDict, instead of\n                                 unordered dict (default False)\n        use_tuple (bool): unpacks arrays into tuples, instead of lists (default\n                          False)\n        allow_invalid_utf8 (bool): unpack invalid strings into instances of\n                                   InvalidString, for access to the bytes\n                                   (default False)\n\n    Returns:\n        A Python object.\n\n    Raises:\n        InsufficientDataException(UnpackException):\n            Insufficient data to unpack the serialized object.\n        InvalidStringException(UnpackException):\n            Invalid UTF-8 string encountered during unpacking.\n        UnsupportedTimestampException(UnpackException):\n            Unsupported timestamp format encountered during unpacking.\n        ReservedCodeException(UnpackException):\n            Reserved code encountered during unpacking.\n        UnhashableKeyException(UnpackException):\n            Unhashable key encountered during map unpacking.\n            The serialized map cannot be deserialized into a Python dictionary.\n        DuplicateKeyException(UnpackException):\n            Duplicate key encountered during map unpacking.\n\n    Example:\n    >>> f = open('test.bin', 'rb')\n    >>> umsgpack.unpackb(f)\n    {'compact': True, 'schema': 0}\n    >>>\n    "
    return _unpack(fp, options)

def _unpackb2(s, **options):
    if False:
        for i in range(10):
            print('nop')
    "\n    Deserialize MessagePack bytes into a Python object.\n\n    Args:\n        s: a 'str' or 'bytearray' containing serialized MessagePack bytes\n\n    Kwargs:\n        ext_handlers (dict): dictionary of Ext handlers, mapping integer Ext\n                             type to a callable that unpacks an instance of\n                             Ext into an object\n        use_ordered_dict (bool): unpack maps into OrderedDict, instead of\n                                 unordered dict (default False)\n        use_tuple (bool): unpacks arrays into tuples, instead of lists (default\n                          False)\n        allow_invalid_utf8 (bool): unpack invalid strings into instances of\n                                   InvalidString, for access to the bytes\n                                   (default False)\n\n    Returns:\n        A Python object.\n\n    Raises:\n        TypeError:\n            Packed data type is neither 'str' nor 'bytearray'.\n        InsufficientDataException(UnpackException):\n            Insufficient data to unpack the serialized object.\n        InvalidStringException(UnpackException):\n            Invalid UTF-8 string encountered during unpacking.\n        UnsupportedTimestampException(UnpackException):\n            Unsupported timestamp format encountered during unpacking.\n        ReservedCodeException(UnpackException):\n            Reserved code encountered during unpacking.\n        UnhashableKeyException(UnpackException):\n            Unhashable key encountered during map unpacking.\n            The serialized map cannot be deserialized into a Python dictionary.\n        DuplicateKeyException(UnpackException):\n            Duplicate key encountered during map unpacking.\n\n    Example:\n    >>> umsgpack.unpackb(b'\x82§compactÃ¦schema\x00')\n    {u'compact': True, u'schema': 0}\n    >>>\n    "
    if not isinstance(s, (str, bytearray)):
        raise TypeError("packed data must be type 'str' or 'bytearray'")
    return _unpack(io.BytesIO(s), options)

def _unpackb3(s, **options):
    if False:
        return 10
    "\n    Deserialize MessagePack bytes into a Python object.\n\n    Args:\n        s: a 'bytes' or 'bytearray' containing serialized MessagePack bytes\n\n    Kwargs:\n        ext_handlers (dict): dictionary of Ext handlers, mapping integer Ext\n                             type to a callable that unpacks an instance of\n                             Ext into an object\n        use_ordered_dict (bool): unpack maps into OrderedDict, instead of\n                                 unordered dict (default False)\n        use_tuple (bool): unpacks arrays into tuples, instead of lists (default\n                          False)\n        allow_invalid_utf8 (bool): unpack invalid strings into instances of\n                                   InvalidString, for access to the bytes\n                                   (default False)\n\n    Returns:\n        A Python object.\n\n    Raises:\n        TypeError:\n            Packed data type is neither 'bytes' nor 'bytearray'.\n        InsufficientDataException(UnpackException):\n            Insufficient data to unpack the serialized object.\n        InvalidStringException(UnpackException):\n            Invalid UTF-8 string encountered during unpacking.\n        UnsupportedTimestampException(UnpackException):\n            Unsupported timestamp format encountered during unpacking.\n        ReservedCodeException(UnpackException):\n            Reserved code encountered during unpacking.\n        UnhashableKeyException(UnpackException):\n            Unhashable key encountered during map unpacking.\n            The serialized map cannot be deserialized into a Python dictionary.\n        DuplicateKeyException(UnpackException):\n            Duplicate key encountered during map unpacking.\n\n    Example:\n    >>> umsgpack.unpackb(b'\x82§compactÃ¦schema\x00')\n    {'compact': True, 'schema': 0}\n    >>>\n    "
    if not isinstance(s, (bytes, bytearray)):
        raise TypeError("packed data must be type 'bytes' or 'bytearray'")
    return _unpack(io.BytesIO(s), options)

def __init():
    if False:
        print('Hello World!')
    global pack
    global packb
    global unpack
    global unpackb
    global dump
    global dumps
    global load
    global loads
    global compatibility
    global _epoch
    global _utc_tzinfo
    global _float_precision
    global _unpack_dispatch_table
    global xrange
    compatibility = False
    if sys.version_info[0] == 3:
        _utc_tzinfo = datetime.timezone.utc
    else:

        class UTC(datetime.tzinfo):
            ZERO = datetime.timedelta(0)

            def utcoffset(self, dt):
                if False:
                    print('Hello World!')
                return UTC.ZERO

            def tzname(self, dt):
                if False:
                    return 10
                return 'UTC'

            def dst(self, dt):
                if False:
                    while True:
                        i = 10
                return UTC.ZERO
        _utc_tzinfo = UTC()
    _epoch = datetime.datetime(1970, 1, 1, tzinfo=_utc_tzinfo)
    if sys.float_info.mant_dig == 53:
        _float_precision = 'double'
    else:
        _float_precision = 'single'
    if sys.version_info[0] == 3:
        pack = _pack3
        packb = _packb3
        dump = _pack3
        dumps = _packb3
        unpack = _unpack3
        unpackb = _unpackb3
        load = _unpack3
        loads = _unpackb3
        xrange = range
    else:
        pack = _pack2
        packb = _packb2
        dump = _pack2
        dumps = _packb2
        unpack = _unpack2
        unpackb = _unpackb2
        load = _unpack2
        loads = _unpackb2
    _unpack_dispatch_table = {}
    for code in range(0, 127 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_integer
    for code in range(128, 143 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_map
    for code in range(144, 159 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_array
    for code in range(160, 191 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_string
    _unpack_dispatch_table[b'\xc0'] = _unpack_nil
    _unpack_dispatch_table[b'\xc1'] = _unpack_reserved
    _unpack_dispatch_table[b'\xc2'] = _unpack_boolean
    _unpack_dispatch_table[b'\xc3'] = _unpack_boolean
    for code in range(196, 198 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_binary
    for code in range(199, 201 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_ext
    _unpack_dispatch_table[b'\xca'] = _unpack_float
    _unpack_dispatch_table[b'\xcb'] = _unpack_float
    for code in range(204, 207 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_integer
    for code in range(208, 211 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_integer
    for code in range(212, 216 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_ext
    for code in range(217, 219 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_string
    _unpack_dispatch_table[b'\xdc'] = _unpack_array
    _unpack_dispatch_table[b'\xdd'] = _unpack_array
    _unpack_dispatch_table[b'\xde'] = _unpack_map
    _unpack_dispatch_table[b'\xdf'] = _unpack_map
    for code in range(224, 255 + 1):
        _unpack_dispatch_table[struct.pack('B', code)] = _unpack_integer
__init()