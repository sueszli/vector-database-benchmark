import sys
from ctypes import *
_array_type = type(Array)

def _other_endian(typ):
    if False:
        print('Hello World!')
    "Return the type with the 'other' byte order.  Simple types like\n    c_int and so on already have __ctype_be__ and __ctype_le__\n    attributes which contain the types, for more complicated types\n    arrays and structures are supported.\n    "
    if hasattr(typ, _OTHER_ENDIAN):
        return getattr(typ, _OTHER_ENDIAN)
    if isinstance(typ, _array_type):
        return _other_endian(typ._type_) * typ._length_
    if issubclass(typ, Structure):
        return typ
    raise TypeError('This type does not support other endian: %s' % typ)

class _swapped_meta(type(Structure)):

    def __setattr__(self, attrname, value):
        if False:
            for i in range(10):
                print('nop')
        if attrname == '_fields_':
            fields = []
            for desc in value:
                name = desc[0]
                typ = desc[1]
                rest = desc[2:]
                fields.append((name, _other_endian(typ)) + rest)
            value = fields
        super().__setattr__(attrname, value)
if sys.byteorder == 'little':
    _OTHER_ENDIAN = '__ctype_be__'
    LittleEndianStructure = Structure

    class BigEndianStructure(Structure, metaclass=_swapped_meta):
        """Structure with big endian byte order"""
        __slots__ = ()
        _swappedbytes_ = None
elif sys.byteorder == 'big':
    _OTHER_ENDIAN = '__ctype_le__'
    BigEndianStructure = Structure

    class LittleEndianStructure(Structure, metaclass=_swapped_meta):
        """Structure with little endian byte order"""
        __slots__ = ()
        _swappedbytes_ = None
else:
    raise RuntimeError('Invalid byteorder')