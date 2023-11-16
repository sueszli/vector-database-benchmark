"""Constants and static functions to support protocol buffer wire format."""
__author__ = 'robinson@google.com (Will Robinson)'
import struct
from google.protobuf import descriptor
from google.protobuf import message
TAG_TYPE_BITS = 3
TAG_TYPE_MASK = (1 << TAG_TYPE_BITS) - 1
WIRETYPE_VARINT = 0
WIRETYPE_FIXED64 = 1
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3
WIRETYPE_END_GROUP = 4
WIRETYPE_FIXED32 = 5
_WIRETYPE_MAX = 5
INT32_MAX = int((1 << 31) - 1)
INT32_MIN = int(-(1 << 31))
UINT32_MAX = (1 << 32) - 1
INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
UINT64_MAX = (1 << 64) - 1
FORMAT_UINT32_LITTLE_ENDIAN = '<I'
FORMAT_UINT64_LITTLE_ENDIAN = '<Q'
FORMAT_FLOAT_LITTLE_ENDIAN = '<f'
FORMAT_DOUBLE_LITTLE_ENDIAN = '<d'
if struct.calcsize(FORMAT_UINT32_LITTLE_ENDIAN) != 4:
    raise AssertionError('Format "I" is not a 32-bit number.')
if struct.calcsize(FORMAT_UINT64_LITTLE_ENDIAN) != 8:
    raise AssertionError('Format "Q" is not a 64-bit number.')

def PackTag(field_number, wire_type):
    if False:
        while True:
            i = 10
    'Returns an unsigned 32-bit integer that encodes the field number and\n  wire type information in standard protocol message wire format.\n\n  Args:\n    field_number: Expected to be an integer in the range [1, 1 << 29)\n    wire_type: One of the WIRETYPE_* constants.\n  '
    if not 0 <= wire_type <= _WIRETYPE_MAX:
        raise message.EncodeError('Unknown wire type: %d' % wire_type)
    return field_number << TAG_TYPE_BITS | wire_type

def UnpackTag(tag):
    if False:
        while True:
            i = 10
    'The inverse of PackTag().  Given an unsigned 32-bit number,\n  returns a (field_number, wire_type) tuple.\n  '
    return (tag >> TAG_TYPE_BITS, tag & TAG_TYPE_MASK)

def ZigZagEncode(value):
    if False:
        return 10
    'ZigZag Transform:  Encodes signed integers so that they can be\n  effectively used with varint encoding.  See wire_format.h for\n  more details.\n  '
    if value >= 0:
        return value << 1
    return value << 1 ^ ~0

def ZigZagDecode(value):
    if False:
        i = 10
        return i + 15
    'Inverse of ZigZagEncode().'
    if not value & 1:
        return value >> 1
    return value >> 1 ^ ~0

def Int32ByteSize(field_number, int32):
    if False:
        return 10
    return Int64ByteSize(field_number, int32)

def Int32ByteSizeNoTag(int32):
    if False:
        print('Hello World!')
    return _VarUInt64ByteSizeNoTag(18446744073709551615 & int32)

def Int64ByteSize(field_number, int64):
    if False:
        i = 10
        return i + 15
    return UInt64ByteSize(field_number, 18446744073709551615 & int64)

def UInt32ByteSize(field_number, uint32):
    if False:
        for i in range(10):
            print('nop')
    return UInt64ByteSize(field_number, uint32)

def UInt64ByteSize(field_number, uint64):
    if False:
        for i in range(10):
            print('nop')
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(uint64)

def SInt32ByteSize(field_number, int32):
    if False:
        i = 10
        return i + 15
    return UInt32ByteSize(field_number, ZigZagEncode(int32))

def SInt64ByteSize(field_number, int64):
    if False:
        while True:
            i = 10
    return UInt64ByteSize(field_number, ZigZagEncode(int64))

def Fixed32ByteSize(field_number, fixed32):
    if False:
        while True:
            i = 10
    return TagByteSize(field_number) + 4

def Fixed64ByteSize(field_number, fixed64):
    if False:
        print('Hello World!')
    return TagByteSize(field_number) + 8

def SFixed32ByteSize(field_number, sfixed32):
    if False:
        return 10
    return TagByteSize(field_number) + 4

def SFixed64ByteSize(field_number, sfixed64):
    if False:
        for i in range(10):
            print('nop')
    return TagByteSize(field_number) + 8

def FloatByteSize(field_number, flt):
    if False:
        return 10
    return TagByteSize(field_number) + 4

def DoubleByteSize(field_number, double):
    if False:
        return 10
    return TagByteSize(field_number) + 8

def BoolByteSize(field_number, b):
    if False:
        while True:
            i = 10
    return TagByteSize(field_number) + 1

def EnumByteSize(field_number, enum):
    if False:
        for i in range(10):
            print('nop')
    return UInt32ByteSize(field_number, enum)

def StringByteSize(field_number, string):
    if False:
        return 10
    return BytesByteSize(field_number, string.encode('utf-8'))

def BytesByteSize(field_number, b):
    if False:
        print('Hello World!')
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(len(b)) + len(b)

def GroupByteSize(field_number, message):
    if False:
        return 10
    return 2 * TagByteSize(field_number) + message.ByteSize()

def MessageByteSize(field_number, message):
    if False:
        for i in range(10):
            print('nop')
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(message.ByteSize()) + message.ByteSize()

def MessageSetItemByteSize(field_number, msg):
    if False:
        print('Hello World!')
    total_size = 2 * TagByteSize(1) + TagByteSize(2) + TagByteSize(3)
    total_size += _VarUInt64ByteSizeNoTag(field_number)
    message_size = msg.ByteSize()
    total_size += _VarUInt64ByteSizeNoTag(message_size)
    total_size += message_size
    return total_size

def TagByteSize(field_number):
    if False:
        for i in range(10):
            print('nop')
    'Returns the bytes required to serialize a tag with this field number.'
    return _VarUInt64ByteSizeNoTag(PackTag(field_number, 0))

def _VarUInt64ByteSizeNoTag(uint64):
    if False:
        for i in range(10):
            print('nop')
    'Returns the number of bytes required to serialize a single varint\n  using boundary value comparisons. (unrolled loop optimization -WPierce)\n  uint64 must be unsigned.\n  '
    if uint64 <= 127:
        return 1
    if uint64 <= 16383:
        return 2
    if uint64 <= 2097151:
        return 3
    if uint64 <= 268435455:
        return 4
    if uint64 <= 34359738367:
        return 5
    if uint64 <= 4398046511103:
        return 6
    if uint64 <= 562949953421311:
        return 7
    if uint64 <= 72057594037927935:
        return 8
    if uint64 <= 9223372036854775807:
        return 9
    if uint64 > UINT64_MAX:
        raise message.EncodeError('Value out of range: %d' % uint64)
    return 10
NON_PACKABLE_TYPES = (descriptor.FieldDescriptor.TYPE_STRING, descriptor.FieldDescriptor.TYPE_GROUP, descriptor.FieldDescriptor.TYPE_MESSAGE, descriptor.FieldDescriptor.TYPE_BYTES)

def IsTypePackable(field_type):
    if False:
        i = 10
        return i + 15
    'Return true iff packable = true is valid for fields of this type.\n\n  Args:\n    field_type: a FieldDescriptor::Type value.\n\n  Returns:\n    True iff fields of this type are packable.\n  '
    return field_type not in NON_PACKABLE_TYPES