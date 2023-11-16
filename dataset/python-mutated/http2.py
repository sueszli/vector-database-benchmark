"""
http2

HTTP/2 support for Scapy
see RFC7540 and RFC7541 for more information

Implements packets and fields required to encode/decode HTTP/2 Frames
and HPack encoded headers
"""
import abc
import re
from io import BytesIO
import struct
from scapy.compat import raw, plain_str, hex_bytes, orb, chb, bytes_encode
from typing import Optional, List, Union, Callable, Any, Tuple, Sized, Pattern
from scapy.base_classes import Packet_metaclass
import scapy.fields as fields
import scapy.packet as packet
import scapy.config as config
import scapy.volatile as volatile
import scapy.error as error

class HPackMagicBitField(fields.BitField):
    """ HPackMagicBitField is a BitField variant that cannot be assigned another
    value than the default one. This field must not be used where there is
    potential for fuzzing. OTOH, this field makes sense (for instance, if the
    magic bits are used by a dispatcher to select the payload class)
    """
    __slots__ = ['_magic']

    def __init__(self, name, default, size):
        if False:
            while True:
                i = 10
        '\n        :param str name: this field instance name.\n        :param int default: this field only valid value.\n        :param int size: this bitfield bitlength.\n        :return: None\n        :raises: AssertionError\n        '
        assert default >= 0
        assert size != 0
        self._magic = default
        super(HPackMagicBitField, self).__init__(name, default, size)

    def addfield(self, pkt, s, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused.  # noqa: E501\n        :param str|(str, int, long) s: either a str if 0 == size%8 or a tuple with the string to add this field to, the  # noqa: E501\n          number of bits already generated and the generated value so far.\n        :param int val: unused; must be equal to default value\n        :return: str|(str, int, long): the s string extended with this field machine representation  # noqa: E501\n        :raises: AssertionError\n        '
        assert val == self._magic, 'val parameter must value {}; received: {}'.format(self._magic, val)
        return super(HPackMagicBitField, self).addfield(pkt, s, self._magic)

    def getfield(self, pkt, s):
        if False:
            print('Hello World!')
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused.  # noqa: E501\n        :param str|(str, int) s: either a str if size%8==0 or a tuple with the string to parse from and the number of  # noqa: E501\n          bits already consumed by previous bitfield-compatible fields.\n        :return: (str|(str, int), int): Returns the remaining string and the parsed value. May return a tuple if there  # noqa: E501\n          are remaining bits to parse in the first byte. Returned value is equal to default value  # noqa: E501\n        :raises: AssertionError\n        '
        r = super(HPackMagicBitField, self).getfield(pkt, s)
        assert isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], int), 'Second element of BitField.getfield return value expected to be an int or a long; API change detected'
        assert r[1] == self._magic, 'Invalid value parsed from s; error in class guessing detected!'
        return r

    def h2i(self, pkt, x):
        if False:
            i = 10
            return i + 15
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused  # noqa: E501\n        :param int x: unused; must be equal to default value\n        :return: int; default value\n        :raises: AssertionError\n        '
        assert x == self._magic, 'EINVAL: x: This field is magic. Do not attempt to modify it. Expected value: {}'.format(self._magic)
        return super(HPackMagicBitField, self).h2i(pkt, self._magic)

    def i2h(self, pkt, x):
        if False:
            while True:
                i = 10
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused  # noqa: E501\n        :param int x: unused; must be equal to default value\n        :return: int; default value\n        :raises: AssertionError\n        '
        assert x == self._magic, 'EINVAL: x: This field is magic. Do not attempt to modify it. Expected value: {}'.format(self._magic)
        return super(HPackMagicBitField, self).i2h(pkt, self._magic)

    def m2i(self, pkt, x):
        if False:
            return 10
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused  # noqa: E501\n        :param int x: must be the machine representatino of the default value\n        :return: int; default value\n        :raises: AssertionError\n        '
        r = super(HPackMagicBitField, self).m2i(pkt, x)
        assert r == self._magic, 'Invalid value parsed from m2i; error in class guessing detected!'
        return r

    def i2m(self, pkt, x):
        if False:
            print('Hello World!')
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused  # noqa: E501\n        :param int x: unused; must be equal to default value\n        :return: int; default value\n        :raises: AssertionError\n        '
        assert x == self._magic, 'EINVAL: x: This field is magic. Do not attempt to modify it. Expected value: {}'.format(self._magic)
        return super(HPackMagicBitField, self).i2m(pkt, self._magic)

    def any2i(self, pkt, x):
        if False:
            while True:
                i = 10
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused  # noqa: E501\n        :param int x: unused; must be equal to default value\n        :return: int; default value\n        :raises: AssertionError\n        '
        assert x == self._magic, 'EINVAL: x: This field is magic. Do not attempt to modify it. Expected value: {}'.format(self._magic)
        return super(HPackMagicBitField, self).any2i(pkt, self._magic)

class AbstractUVarIntField(fields.Field):
    """AbstractUVarIntField represents an integer as defined in RFC7541
    """
    __slots__ = ['_max_value', 'size', 'rev']
    '\n    :var int size: the bit length of the prefix of this AbstractUVarIntField. It  # noqa: E501\n        represents the complement of the number of MSB that are used in the\n        current byte for other purposes by some other BitFields\n    :var int _max_value: the maximum value that can be stored in the\n        sole prefix. If the integer equals or exceeds this value, the max prefix\n        value is assigned to the size first bits and the multibyte representation\n        is used\n    :var bool rev: is a fake property, also emulated for the sake of\n        compatibility with Bitfields\n    '

    def __init__(self, name, default, size):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param str name: the name of this field instance\n        :param int|None default: positive, null or None default value for this field instance.  # noqa: E501\n        :param int size: the number of bits to consider in the first byte. Valid range is ]0;8]  # noqa: E501\n        :return: None\n        :raises: AssertionError\n        '
        assert default is None or (isinstance(default, int) and default >= 0)
        assert 0 < size <= 8
        super(AbstractUVarIntField, self).__init__(name, default)
        self.size = size
        self._max_value = (1 << self.size) - 1
        self.rev = False

    def h2i(self, pkt, x):
        if False:
            return 10
        '\n        :param packet.Packet|None pkt: unused.\n        :param int|None x: the value to convert.\n        :return: int|None: the converted value.\n        :raises: AssertionError\n        '
        assert not isinstance(x, int) or x >= 0
        return x

    def i2h(self, pkt, x):
        if False:
            return 10
        '\n        :param packet.Packet|None pkt: unused.\n        :param int|None x: the value to convert.\n        :return:: int|None: the converted value.\n        '
        return x

    def _detect_multi_byte(self, fb):
        if False:
            return 10
        ' _detect_multi_byte returns whether the AbstractUVarIntField is represented on  # noqa: E501\n          multiple bytes or not.\n\n          A multibyte representation is indicated by all of the first size bits being set  # noqa: E501\n\n        :param str fb: first byte, as a character.\n        :return: bool: True if multibyte repr detected, else False.\n        :raises: AssertionError\n        '
        assert isinstance(fb, int) or len(fb) == 1
        return orb(fb) & self._max_value == self._max_value

    def _parse_multi_byte(self, s):
        if False:
            print('Hello World!')
        ' _parse_multi_byte parses x as a multibyte representation to get the\n          int value of this AbstractUVarIntField.\n\n        :param str s: the multibyte string to parse.\n        :return: int: The parsed int value represented by this AbstractUVarIntField.  # noqa: E501\n        :raises:: AssertionError\n        :raises:: Scapy_Exception if the input value encodes an integer larger than 1<<64  # noqa: E501\n        '
        assert len(s) >= 2
        tmp_len = len(s)
        value = 0
        i = 1
        byte = orb(s[i])
        max_value = 1 << 64
        while byte & 128:
            value += (byte ^ 128) << 7 * (i - 1)
            if value > max_value:
                raise error.Scapy_Exception('out-of-bound value: the string encodes a value that is too large (>2^{{64}}): {}'.format(value))
            i += 1
            assert i < tmp_len, 'EINVAL: x: out-of-bound read: the string ends before the AbstractUVarIntField!'
            byte = orb(s[i])
        value += byte << 7 * (i - 1)
        value += self._max_value
        assert value >= 0
        return value

    def m2i(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        '\n          A tuple is expected for the "x" param only if "size" is different than 8. If a tuple is received, some bits  # noqa: E501\n          were consumed by another field. This field consumes the remaining bits, therefore the int of the tuple must  # noqa: E501\n          equal "size".\n\n        :param packet.Packet|None pkt: unused.\n        :param str|(str, int) x: the string to convert. If bits were consumed by a previous bitfield-compatible field.  # noqa: E501\n        :raises: AssertionError\n        '
        assert isinstance(x, bytes) or (isinstance(x, tuple) and x[1] >= 0)
        if isinstance(x, tuple):
            assert 8 - x[1] == self.size, 'EINVAL: x: not enough bits remaining in current byte to read the prefix'
            val = x[0]
        else:
            assert isinstance(x, bytes) and self.size == 8, 'EINVAL: x: tuple expected when prefix_len is not a full byte'
            val = x
        if self._detect_multi_byte(val[0]):
            ret = self._parse_multi_byte(val)
        else:
            ret = orb(val[0]) & self._max_value
        assert ret >= 0
        return ret

    def i2m(self, pkt, x):
        if False:
            return 10
        '\n        :param packet.Packet|None pkt: unused.\n        :param int x: the value to convert.\n        :return: str: the converted value.\n        :raises: AssertionError\n        '
        assert x >= 0
        if x < self._max_value:
            return chb(x)
        else:
            sl = [chb(self._max_value)]
            x -= self._max_value
            while x >= 128:
                sl.append(chb(128 | x & 127))
                x >>= 7
            sl.append(chb(x))
            return b''.join(sl)

    def any2i(self, pkt, x):
        if False:
            return 10
        '\n          A "x" value as a string is parsed as a binary encoding of a UVarInt. An int is considered an internal value.  # noqa: E501\n          None is returned as is.\n\n        :param packet.Packet|None pkt: the packet containing this field; probably unused.  # noqa: E501\n        :param str|int|None x: the value to convert.\n        :return: int|None: the converted value.\n        :raises: AssertionError\n        '
        if isinstance(x, type(None)):
            return x
        if isinstance(x, int):
            assert x >= 0
            ret = self.h2i(pkt, x)
            assert isinstance(ret, int) and ret >= 0
            return ret
        elif isinstance(x, bytes):
            ret = self.m2i(pkt, x)
            assert isinstance(ret, int) and ret >= 0
            return ret
        assert False, 'EINVAL: x: No idea what the parameter format is'

    def i2repr(self, pkt, x):
        if False:
            while True:
                i = 10
        '\n        :param packet.Packet|None pkt: probably unused.\n        :param x: int|None: the positive, null or none value to convert.\n        :return: str: the representation of the value.\n        '
        return repr(self.i2h(pkt, x))

    def addfield(self, pkt, s, val):
        if False:
            return 10
        '\n          An AbstractUVarIntField prefix always consumes the remaining bits\n          of a BitField;if no current BitField is in use (no tuple in\n          entry) then the prefix length is 8 bits and the whole byte is to\n          be consumed\n\n        :param packet.Packet|None pkt: the packet containing this field.\n          Probably unused.\n        :param str|(str, int, long) s: the string to append this field to.\n          A tuple indicates that some bits were already generated by another\n          bitfield-compatible field. This MUST be the case if "size" is not 8.\n          The int is the number of bits already generated in the first byte of\n          the str. The long is the value that was generated by the previous\n          bitfield-compatible fields.\n        :param int val: the positive or null value to be added.\n        :return: str: s concatenated with the machine representation of this\n          field.\n        :raises: AssertionError\n        '
        assert val >= 0
        if isinstance(s, bytes):
            assert self.size == 8, 'EINVAL: s: tuple expected when prefix_len is not a full byte'
            return s + self.i2m(pkt, val)
        if val >= self._max_value:
            return s[0] + chb((s[2] << self.size) + self._max_value) + self.i2m(pkt, val)[1:]
        return s[0] + chb((s[2] << self.size) + orb(self.i2m(pkt, val)))

    @staticmethod
    def _detect_bytelen_from_str(s):
        if False:
            print('Hello World!')
        ' _detect_bytelen_from_str returns the length of the machine\n          representation of an AbstractUVarIntField starting at the beginning\n          of s and which is assumed to expand over multiple bytes\n          (value > _max_prefix_value).\n\n        :param str s: the string to parse. It is assumed that it is a multibyte int.  # noqa: E501\n        :return: The bytelength of the AbstractUVarIntField.\n        :raises: AssertionError\n        '
        assert len(s) >= 2
        tmp_len = len(s)
        i = 1
        while orb(s[i]) & 128 > 0:
            i += 1
            assert i < tmp_len, 'EINVAL: s: out-of-bound read: unfinished AbstractUVarIntField detected'
        ret = i + 1
        assert ret >= 0
        return ret

    def i2len(self, pkt, x):
        if False:
            while True:
                i = 10
        '\n        :param packet.Packet|None pkt: unused.\n        :param int x: the positive or null value whose binary size if requested.  # noqa: E501\n        :raises: AssertionError\n        '
        assert x >= 0
        if x < self._max_value:
            return 1
        x -= self._max_value
        i = 1
        if x == 0:
            i += 1
        while x > 0:
            x >>= 7
            i += 1
        ret = i
        assert ret >= 0
        return ret

    def getfield(self, pkt, s):
        if False:
            while True:
                i = 10
        '\n        :param packet.Packet|None pkt: the packet instance containing this\n          field; probably unused.\n        :param str|(str, int) s: the input value to get this field value from.\n          If size is 8, s is a string, else it is a tuple containing the value\n          and an int indicating the number of bits already consumed in the\n          first byte of the str. The number of remaining bits to consume in the\n          first byte must be equal to "size".\n        :return: (str, int): the remaining bytes of s and the parsed value.\n        :raises: AssertionError\n        '
        if isinstance(s, tuple):
            assert len(s) == 2
            temp = s
            (ts, ti) = temp
            assert ti >= 0
            assert 8 - ti == self.size, 'EINVAL: s: not enough bits remaining in current byte to read the prefix'
            val = ts
        else:
            assert isinstance(s, bytes) and self.size == 8, 'EINVAL: s: tuple expected when prefix_len is not a full byte'
            val = s
        if self._detect_multi_byte(val[0]):
            tmp_len = self._detect_bytelen_from_str(val)
        else:
            tmp_len = 1
        ret = (val[tmp_len:], self.m2i(pkt, s))
        assert ret[1] >= 0
        return ret

    def randval(self):
        if False:
            return 10
        '\n        :return: volatile.VolatileValue: a volatile value for this field "long"-compatible internal value.  # noqa: E501\n        '
        return volatile.RandLong()

class UVarIntField(AbstractUVarIntField):

    def __init__(self, name, default, size):
        if False:
            print('Hello World!')
        '\n        :param str name: the name of this field instance.\n        :param default: the default value for this field instance. default must be positive or null.  # noqa: E501\n        :raises: AssertionError\n        '
        assert default >= 0
        assert 0 < size <= 8
        super(UVarIntField, self).__init__(name, default, size)
        self.size = size
        self._max_value = (1 << self.size) - 1
        self.rev = False

    def h2i(self, pkt, x):
        if False:
            i = 10
            return i + 15
        ' h2i is overloaded to restrict the acceptable x values (not None)\n\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused.  # noqa: E501\n        :param int x: the value to convert.\n        :return: int: the converted value.\n        :raises: AssertionError\n        '
        ret = super(UVarIntField, self).h2i(pkt, x)
        assert not isinstance(ret, type(None)) and ret >= 0
        return ret

    def i2h(self, pkt, x):
        if False:
            print('Hello World!')
        ' i2h is overloaded to restrict the acceptable x values (not None)\n\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused.  # noqa: E501\n        :param int x: the value to convert.\n        :return: int: the converted value.\n        :raises: AssertionError\n        '
        ret = super(UVarIntField, self).i2h(pkt, x)
        assert not isinstance(ret, type(None)) and ret >= 0
        return ret

    def any2i(self, pkt, x):
        if False:
            print('Hello World!')
        ' any2i is overloaded to restrict the acceptable x values (not None)\n\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused.  # noqa: E501\n        :param str|int x: the value to convert.\n        :return: int: the converted value.\n        :raises: AssertionError\n        '
        ret = super(UVarIntField, self).any2i(pkt, x)
        assert not isinstance(ret, type(None)) and ret >= 0
        return ret

    def i2repr(self, pkt, x):
        if False:
            return 10
        ' i2repr is overloaded to restrict the acceptable x values (not None)\n\n        :param packet.Packet|None pkt: the packet instance containing this field instance; probably unused.  # noqa: E501\n        :param int x: the value to convert.\n        :return: str: the converted value.\n        '
        return super(UVarIntField, self).i2repr(pkt, x)

class FieldUVarLenField(AbstractUVarIntField):
    __slots__ = ['_length_of', '_adjust']

    def __init__(self, name, default, size, length_of, adjust=lambda x: x):
        if False:
            while True:
                i = 10
        ' Initializes a FieldUVarLenField\n\n        :param str name: The name of this field instance.\n        :param int|None default: the default value of this field instance.\n        :param int size: the number of bits that are occupied by this field in the first byte of a binary string.  # noqa: E501\n          size must be in the range ]0;8].\n        :param str length_of: The name of the field this field value is measuring/representing.  # noqa: E501\n        :param callable adjust: A function that modifies the value computed from the "length_of" field.  # noqa: E501\n\n        adjust can be used for instance to add a constant to the length_of field  # noqa: E501\n         length. For instance, let\'s say that i2len of the length_of field\n         returns 2. If adjust is lambda x: x+1 In that case, this field will\n         value 3 at build time.\n        :return: None\n        :raises: AssertionError\n        '
        assert default is None or default >= 0
        assert 0 < size <= 8
        super(FieldUVarLenField, self).__init__(name, default, size)
        self._length_of = length_of
        self._adjust = adjust

    def addfield(self, pkt, s, val):
        if False:
            while True:
                i = 10
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance. This parameter must not be  # noqa: E501\n          None if the val parameter is.\n        :param str|(str, int, long) s: the string to append this field to. A tuple indicates that some bits were already  # noqa: E501\n          generated by another bitfield-compatible field. This MUST be the case if "size" is not 8. The int is the  # noqa: E501\n          number of bits already generated in the first byte of the str. The long is the value that was generated by the  # noqa: E501\n          previous bitfield-compatible fields.\n        :param int|None val: the positive or null value to be added. If None, the value is computed from pkt.  # noqa: E501\n        :return: str: s concatenated with the machine representation of this field.  # noqa: E501\n        :raises: AssertionError\n        '
        if val is None:
            assert isinstance(pkt, packet.Packet), 'EINVAL: pkt: Packet expected when val is None; received {}'.format(type(pkt))
            val = self._compute_value(pkt)
        return super(FieldUVarLenField, self).addfield(pkt, s, val)

    def i2m(self, pkt, x):
        if False:
            print('Hello World!')
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance. This parameter must not be  # noqa: E501\n          None if the x parameter is.\n        :param int|None x: the positive or null value to be added. If None, the value is computed from pkt.  # noqa: E501\n        :return: str\n        :raises: AssertionError\n        '
        if x is None:
            assert isinstance(pkt, packet.Packet), 'EINVAL: pkt: Packet expected when x is None; received {}'.format(type(pkt))
            x = self._compute_value(pkt)
        return super(FieldUVarLenField, self).i2m(pkt, x)

    def _compute_value(self, pkt):
        if False:
            i = 10
            return i + 15
        ' Computes the value of this field based on the provided packet and\n        the length_of field and the adjust callback\n\n        :param packet.Packet pkt: the packet from which is computed this field value.  # noqa: E501\n        :return: int: the computed value for this field.\n        :raises: KeyError: the packet nor its payload do not contain an attribute\n          with the length_of name.\n        :raises: AssertionError\n        :raises: KeyError if _length_of is not one of pkt fields\n        '
        (fld, fval) = pkt.getfield_and_val(self._length_of)
        val = fld.i2len(pkt, fval)
        ret = self._adjust(val)
        assert ret >= 0
        return ret

class HPackStringsInterface(Sized, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __bytes__(self):
        if False:
            print('Hello World!')
        r = self.__str__()
        return bytes_encode(r)

    @abc.abstractmethod
    def origin(self):
        if False:
            i = 10
            return i + 15
        pass

    @abc.abstractmethod
    def __len__(self):
        if False:
            print('Hello World!')
        pass

class HPackLiteralString(HPackStringsInterface):
    """ HPackLiteralString is a string. This class is used as a marker and
    implements an interface in common with HPackZString
    """
    __slots__ = ['_s']

    def __init__(self, s):
        if False:
            print('Hello World!')
        self._s = s

    def __str__(self):
        if False:
            return 10
        return self._s

    def origin(self):
        if False:
            i = 10
            return i + 15
        return plain_str(self._s)

    def __len__(self):
        if False:
            return 10
        return len(self._s)

class EOS(object):
    """ Simple "marker" to designate the End Of String symbol in the huffman table
    """

class HuffmanNode(object):
    """ HuffmanNode is an entry of the binary tree used for encoding/decoding
    HPack compressed HTTP/2 headers
    """
    __slots__ = ['left', 'right']
    '@var l: the left branch of this node\n    @var r: the right branch of this Node\n\n    These variables can value None (leaf node), another HuffmanNode, or a\n     symbol. Symbols are either a character or the End Of String symbol (class\n     EOS)\n    '

    def __init__(self, left, right):
        if False:
            print('Hello World!')
        self.left = left
        self.right = right

    def __getitem__(self, b):
        if False:
            return 10
        return self.right if b else self.left

    def __setitem__(self, b, val):
        if False:
            print('Hello World!')
        if b:
            self.right = val
        else:
            self.left = val

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.__repr__()

    def __repr__(self):
        if False:
            return 10
        return '({}, {})'.format(self.left, self.right)

class InvalidEncodingException(Exception):
    """ InvalidEncodingException is raised when a supposedly huffman-encoded
     string is decoded and a decoding error arises
    """

class HPackZString(HPackStringsInterface):
    __slots__ = ['_s', '_encoded']
    static_huffman_code = [(8184, 13), (8388568, 23), (268435426, 28), (268435427, 28), (268435428, 28), (268435429, 28), (268435430, 28), (268435431, 28), (268435432, 28), (16777194, 24), (1073741820, 30), (268435433, 28), (268435434, 28), (1073741821, 30), (268435435, 28), (268435436, 28), (268435437, 28), (268435438, 28), (268435439, 28), (268435440, 28), (268435441, 28), (268435442, 28), (1073741822, 30), (268435443, 28), (268435444, 28), (268435445, 28), (268435446, 28), (268435447, 28), (268435448, 28), (268435449, 28), (268435450, 28), (268435451, 28), (20, 6), (1016, 10), (1017, 10), (4090, 12), (8185, 13), (21, 6), (248, 8), (2042, 11), (1018, 10), (1019, 10), (249, 8), (2043, 11), (250, 8), (22, 6), (23, 6), (24, 6), (0, 5), (1, 5), (2, 5), (25, 6), (26, 6), (27, 6), (28, 6), (29, 6), (30, 6), (31, 6), (92, 7), (251, 8), (32764, 15), (32, 6), (4091, 12), (1020, 10), (8186, 13), (33, 6), (93, 7), (94, 7), (95, 7), (96, 7), (97, 7), (98, 7), (99, 7), (100, 7), (101, 7), (102, 7), (103, 7), (104, 7), (105, 7), (106, 7), (107, 7), (108, 7), (109, 7), (110, 7), (111, 7), (112, 7), (113, 7), (114, 7), (252, 8), (115, 7), (253, 8), (8187, 13), (524272, 19), (8188, 13), (16380, 14), (34, 6), (32765, 15), (3, 5), (35, 6), (4, 5), (36, 6), (5, 5), (37, 6), (38, 6), (39, 6), (6, 5), (116, 7), (117, 7), (40, 6), (41, 6), (42, 6), (7, 5), (43, 6), (118, 7), (44, 6), (8, 5), (9, 5), (45, 6), (119, 7), (120, 7), (121, 7), (122, 7), (123, 7), (32766, 15), (2044, 11), (16381, 14), (8189, 13), (268435452, 28), (1048550, 20), (4194258, 22), (1048551, 20), (1048552, 20), (4194259, 22), (4194260, 22), (4194261, 22), (8388569, 23), (4194262, 22), (8388570, 23), (8388571, 23), (8388572, 23), (8388573, 23), (8388574, 23), (16777195, 24), (8388575, 23), (16777196, 24), (16777197, 24), (4194263, 22), (8388576, 23), (16777198, 24), (8388577, 23), (8388578, 23), (8388579, 23), (8388580, 23), (2097116, 21), (4194264, 22), (8388581, 23), (4194265, 22), (8388582, 23), (8388583, 23), (16777199, 24), (4194266, 22), (2097117, 21), (1048553, 20), (4194267, 22), (4194268, 22), (8388584, 23), (8388585, 23), (2097118, 21), (8388586, 23), (4194269, 22), (4194270, 22), (16777200, 24), (2097119, 21), (4194271, 22), (8388587, 23), (8388588, 23), (2097120, 21), (2097121, 21), (4194272, 22), (2097122, 21), (8388589, 23), (4194273, 22), (8388590, 23), (8388591, 23), (1048554, 20), (4194274, 22), (4194275, 22), (4194276, 22), (8388592, 23), (4194277, 22), (4194278, 22), (8388593, 23), (67108832, 26), (67108833, 26), (1048555, 20), (524273, 19), (4194279, 22), (8388594, 23), (4194280, 22), (33554412, 25), (67108834, 26), (67108835, 26), (67108836, 26), (134217694, 27), (134217695, 27), (67108837, 26), (16777201, 24), (33554413, 25), (524274, 19), (2097123, 21), (67108838, 26), (134217696, 27), (134217697, 27), (67108839, 26), (134217698, 27), (16777202, 24), (2097124, 21), (2097125, 21), (67108840, 26), (67108841, 26), (268435453, 28), (134217699, 27), (134217700, 27), (134217701, 27), (1048556, 20), (16777203, 24), (1048557, 20), (2097126, 21), (4194281, 22), (2097127, 21), (2097128, 21), (8388595, 23), (4194282, 22), (4194283, 22), (33554414, 25), (33554415, 25), (16777204, 24), (16777205, 24), (67108842, 26), (8388596, 23), (67108843, 26), (134217702, 27), (67108844, 26), (67108845, 26), (134217703, 27), (134217704, 27), (134217705, 27), (134217706, 27), (134217707, 27), (268435454, 28), (134217708, 27), (134217709, 27), (134217710, 27), (134217711, 27), (134217712, 27), (67108846, 26), (1073741823, 30)]
    static_huffman_tree = None

    @classmethod
    def _huffman_encode_char(cls, c):
        if False:
            return 10
        ' huffman_encode_char assumes that the static_huffman_tree was\n        previously initialized\n\n        :param str|EOS c: a symbol to encode\n        :return: (int, int): the bitstring of the symbol and its bitlength\n        :raises: AssertionError\n        '
        if isinstance(c, EOS):
            return cls.static_huffman_code[-1]
        else:
            assert isinstance(c, int) or len(c) == 1
        return cls.static_huffman_code[orb(c)]

    @classmethod
    def huffman_encode(cls, s):
        if False:
            return 10
        ' huffman_encode returns the bitstring and the bitlength of the\n        bitstring representing the string provided as a parameter\n\n        :param str s: the string to encode\n        :return: (int, int): the bitstring of s and its bitlength\n        :raises: AssertionError\n        '
        i = 0
        ibl = 0
        for c in s:
            (val, bl) = cls._huffman_encode_char(c)
            i = (i << bl) + val
            ibl += bl
        padlen = 8 - ibl % 8
        if padlen != 8:
            (val, bl) = cls._huffman_encode_char(EOS())
            i = (i << padlen) + (val >> bl - padlen)
            ibl += padlen
        ret = (i, ibl)
        assert ret[0] >= 0
        assert ret[1] >= 0
        return ret

    @classmethod
    def huffman_decode(cls, i, ibl):
        if False:
            print('Hello World!')
        ' huffman_decode decodes the bitstring provided as parameters.\n\n        :param int i: the bitstring to decode\n        :param int ibl: the bitlength of i\n        :return: str: the string decoded from the bitstring\n        :raises: AssertionError, InvalidEncodingException\n        '
        assert i >= 0
        assert ibl >= 0
        if isinstance(cls.static_huffman_tree, type(None)):
            cls.huffman_compute_decode_tree()
        assert not isinstance(cls.static_huffman_tree, type(None))
        s = []
        j = 0
        interrupted = False
        cur = cls.static_huffman_tree
        cur_sym = 0
        cur_sym_bl = 0
        while j < ibl:
            b = i >> ibl - j - 1 & 1
            cur_sym = (cur_sym << 1) + b
            cur_sym_bl += 1
            elmt = cur[b]
            if isinstance(elmt, HuffmanNode):
                interrupted = True
                cur = elmt
                if isinstance(cur, type(None)):
                    raise AssertionError()
            elif isinstance(elmt, EOS):
                raise InvalidEncodingException('Huffman decoder met the full EOS symbol')
            elif isinstance(elmt, bytes):
                interrupted = False
                s.append(elmt)
                cur = cls.static_huffman_tree
                cur_sym = 0
                cur_sym_bl = 0
            else:
                raise InvalidEncodingException('Should never happen, so incidentally it will')
            j += 1
        if interrupted:
            if cur_sym_bl > 7:
                raise InvalidEncodingException('Huffman decoder is detecting padding longer than 7 bits')
            eos_symbol = cls.static_huffman_code[-1]
            eos_msb = eos_symbol[0] >> eos_symbol[1] - cur_sym_bl
            if eos_msb != cur_sym:
                raise InvalidEncodingException('Huffman decoder is detecting unexpected padding format')
        return b''.join(s)

    @classmethod
    def huffman_conv2str(cls, bit_str, bit_len):
        if False:
            i = 10
            return i + 15
        ' huffman_conv2str converts a bitstring of bit_len bitlength into a\n        binary string. It DOES NOT compress/decompress the bitstring!\n\n        :param int bit_str: the bitstring to convert.\n        :param int bit_len: the bitlength of bit_str.\n        :return: str: the converted bitstring as a bytestring.\n        :raises: AssertionError\n        '
        assert bit_str >= 0
        assert bit_len >= 0
        byte_len = bit_len // 8
        rem_bit = bit_len % 8
        if rem_bit != 0:
            bit_str <<= 8 - rem_bit
            byte_len += 1
        s = []
        i = 0
        while i < byte_len:
            s.insert(0, chb(bit_str >> i * 8 & 255))
            i += 1
        return b''.join(s)

    @classmethod
    def huffman_conv2bitstring(cls, s):
        if False:
            while True:
                i = 10
        ' huffman_conv2bitstring converts a string into its bitstring\n        representation. It returns a tuple: the bitstring and its bitlength.\n        This function DOES NOT compress/decompress the string!\n\n        :param str s: the bytestring to convert.\n        :return: (int, int): the bitstring of s, and its bitlength.\n        :raises: AssertionError\n        '
        i = 0
        ibl = len(s) * 8
        for c in s:
            i = (i << 8) + orb(c)
        ret = (i, ibl)
        assert ret[0] >= 0
        assert ret[1] >= 0
        return ret

    @classmethod
    def huffman_compute_decode_tree(cls):
        if False:
            for i in range(10):
                print('nop')
        ' huffman_compute_decode_tree initializes/builds the static_huffman_tree\n\n        :return: None\n        :raises: InvalidEncodingException if there is an encoding problem\n        '
        cls.static_huffman_tree = HuffmanNode(None, None)
        i = 0
        for entry in cls.static_huffman_code:
            parent = cls.static_huffman_tree
            for idx in range(entry[1] - 1, -1, -1):
                b = entry[0] >> idx & 1
                if isinstance(parent[b], bytes):
                    raise InvalidEncodingException('Huffman unique prefix violation :/')
                if idx == 0:
                    parent[b] = chb(i) if i < 256 else EOS()
                elif parent[b] is None:
                    parent[b] = HuffmanNode(None, None)
                parent = parent[b]
            i += 1

    def __init__(self, s):
        if False:
            print('Hello World!')
        self._s = s
        (i, ibl) = type(self).huffman_encode(s)
        self._encoded = type(self).huffman_conv2str(i, ibl)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._encoded

    def origin(self):
        if False:
            print('Hello World!')
        return plain_str(self._s)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._encoded)

class HPackStrLenField(fields.Field):
    """ HPackStrLenField is a StrLenField variant specialized for HTTP/2 HPack

    This variant uses an internal representation that implements HPackStringsInterface.  # noqa: E501
    """
    __slots__ = ['_length_from', '_type_from']

    def __init__(self, name, default, length_from, type_from):
        if False:
            for i in range(10):
                print('nop')
        super(HPackStrLenField, self).__init__(name, default)
        self._length_from = length_from
        self._type_from = type_from

    def addfield(self, pkt, s, val):
        if False:
            i = 10
            return i + 15
        return s + self.i2m(pkt, val)

    @staticmethod
    def _parse(t, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param bool t: whether this string is a huffman compressed string.\n        :param str s: the string to parse.\n        :return: HPackStringsInterface: either a HPackLiteralString or HPackZString, depending on t.  # noqa: E501\n        :raises: InvalidEncodingException\n        '
        if t:
            (i, ibl) = HPackZString.huffman_conv2bitstring(s)
            return HPackZString(HPackZString.huffman_decode(i, ibl))
        return HPackLiteralString(s)

    def getfield(self, pkt, s):
        if False:
            i = 10
            return i + 15
        '\n        :param packet.Packet pkt: the packet instance containing this field instance.  # noqa: E501\n        :param str s: the string to parse this field from.\n        :return: (str, HPackStringsInterface): the remaining string after this field was carved out & the extracted  # noqa: E501\n          value.\n        :raises: KeyError if "type_from" is not a field of pkt or its payloads.\n        :raises: InvalidEncodingException\n        '
        tmp_len = self._length_from(pkt)
        t = pkt.getfieldval(self._type_from) == 1
        return (s[tmp_len:], self._parse(t, s[:tmp_len]))

    def i2h(self, pkt, x):
        if False:
            while True:
                i = 10
        fmt = ''
        if isinstance(x, HPackLiteralString):
            fmt = 'HPackLiteralString({})'
        elif isinstance(x, HPackZString):
            fmt = 'HPackZString({})'
        return fmt.format(x.origin())

    def h2i(self, pkt, x):
        if False:
            while True:
                i = 10
        return HPackLiteralString(x)

    def m2i(self, pkt, x):
        if False:
            i = 10
            return i + 15
        '\n        :param packet.Packet pkt: the packet instance containing this field instance.  # noqa: E501\n        :param str x: the string to parse.\n        :return: HPackStringsInterface: the internal type of the value parsed from x.  # noqa: E501\n        :raises: AssertionError\n        :raises: InvalidEncodingException\n        :raises: KeyError if _type_from is not one of pkt fields.\n        '
        t = pkt.getfieldval(self._type_from)
        tmp_len = self._length_from(pkt)
        assert t is not None and tmp_len is not None, 'Conversion from string impossible: no type or length specified'
        return self._parse(t == 1, x[:tmp_len])

    def any2i(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param packet.Packet|None pkt: the packet instance containing this field instance.  # noqa: E501\n        :param str|HPackStringsInterface x: the value to convert\n        :return: HPackStringsInterface: the Scapy internal value for this field\n        :raises: AssertionError, InvalidEncodingException\n        '
        if isinstance(x, bytes):
            assert isinstance(pkt, packet.Packet)
            return self.m2i(pkt, x)
        assert isinstance(x, HPackStringsInterface)
        return x

    def i2m(self, pkt, x):
        if False:
            while True:
                i = 10
        return raw(x)

    def i2len(self, pkt, x):
        if False:
            while True:
                i = 10
        return len(x)

    def i2repr(self, pkt, x):
        if False:
            print('Hello World!')
        return repr(self.i2h(pkt, x))

class HPackHdrString(packet.Packet):
    """ HPackHdrString is a packet that that is serialized into a RFC7541 par5.2
    string literal repr.
    """
    name = 'HPack Header String'
    fields_desc = [fields.BitEnumField('type', None, 1, {0: 'Literal', 1: 'Compressed'}), FieldUVarLenField('len', None, 7, length_of='data'), HPackStrLenField('data', HPackLiteralString(''), length_from=lambda pkt: pkt.getfieldval('len'), type_from='type')]

    def guess_payload_class(self, payload):
        if False:
            for i in range(10):
                print('nop')
        return config.conf.padding_layer

    def self_build(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'self_build is overridden because type and len are determined at\n        build time, based on the "data" field internal type\n        '
        if self.getfieldval('type') is None:
            self.type = 1 if isinstance(self.getfieldval('data'), HPackZString) else 0
        return super(HPackHdrString, self).self_build(**kwargs)

class HPackHeaders(packet.Packet):
    """HPackHeaders uses the "dispatch_hook" trick of Packet_metaclass to select
    the correct HPack header packet type. For this, the first byte of the string  # noqa: E501
    to dissect is snooped on.
    """

    @classmethod
    def dispatch_hook(cls, s=None, *_args, **_kwds):
        if False:
            return 10
        'dispatch_hook returns the subclass of HPackHeaders that must be used\n        to dissect the string.\n        '
        if s is None:
            return config.conf.raw_layer
        fb = orb(s[0])
        if fb & 128 != 0:
            return HPackIndexedHdr
        if fb & 64 != 0:
            return HPackLitHdrFldWithIncrIndexing
        if fb & 32 != 0:
            return HPackDynamicSizeUpdate
        return HPackLitHdrFldWithoutIndexing

    def guess_payload_class(self, payload):
        if False:
            while True:
                i = 10
        return config.conf.padding_layer

class HPackIndexedHdr(HPackHeaders):
    """ HPackIndexedHdr implements RFC 7541 par6.1
    """
    name = 'HPack Indexed Header Field'
    fields_desc = [HPackMagicBitField('magic', 1, 1), UVarIntField('index', 2, 7)]

class HPackLitHdrFldWithIncrIndexing(HPackHeaders):
    """ HPackLitHdrFldWithIncrIndexing implements RFC 7541 par6.2.1
    """
    name = 'HPack Literal Header With Incremental Indexing'
    fields_desc = [HPackMagicBitField('magic', 1, 2), UVarIntField('index', 0, 6), fields.ConditionalField(fields.PacketField('hdr_name', None, HPackHdrString), lambda pkt: pkt.getfieldval('index') == 0), fields.PacketField('hdr_value', None, HPackHdrString)]

class HPackLitHdrFldWithoutIndexing(HPackHeaders):
    """ HPackLitHdrFldWithIncrIndexing implements RFC 7541 par6.2.2
    and par6.2.3
    """
    name = 'HPack Literal Header Without Indexing (or Never Indexing)'
    fields_desc = [HPackMagicBitField('magic', 0, 3), fields.BitEnumField('never_index', 0, 1, {0: "Don't Index", 1: 'Never Index'}), UVarIntField('index', 0, 4), fields.ConditionalField(fields.PacketField('hdr_name', None, HPackHdrString), lambda pkt: pkt.getfieldval('index') == 0), fields.PacketField('hdr_value', None, HPackHdrString)]

class HPackDynamicSizeUpdate(HPackHeaders):
    """ HPackDynamicSizeUpdate implements RFC 7541 par6.3
    """
    name = 'HPack Dynamic Size Update'
    fields_desc = [HPackMagicBitField('magic', 1, 3), UVarIntField('max_size', 0, 5)]

class H2FramePayload(packet.Packet):
    """ H2FramePayload is an empty class that is a super class of all Scapy
    HTTP/2 Frame Packets
    """

class H2DataFrame(H2FramePayload):
    """ H2DataFrame implements RFC7540 par6.1
    This packet is the Data Frame to use when there is no padding.
    """
    type_id = 0
    END_STREAM_FLAG = 0
    PADDED_FLAG = 3
    flags = {END_STREAM_FLAG: fields.MultiFlagsEntry('ES', 'End Stream'), PADDED_FLAG: fields.MultiFlagsEntry('P', 'Padded')}
    name = 'HTTP/2 Data Frame'
    fields_desc = [fields.StrField('data', '')]

class H2PaddedDataFrame(H2DataFrame):
    """ H2DataFrame implements RFC7540 par6.1
    This packet is the Data Frame to use when there is padding.
    """
    __slots__ = ['s_len']
    name = 'HTTP/2 Padded Data Frame'
    fields_desc = [fields.FieldLenField('padlen', None, length_of='padding', fmt='B'), fields.StrLenField('data', '', length_from=lambda pkt: pkt.get_data_len()), fields.StrLenField('padding', '', length_from=lambda pkt: pkt.getfieldval('padlen'))]

    def get_data_len(self):
        if False:
            print('Hello World!')
        ' get_data_len computes the length of the data field\n\n        To do this computation, the length of the padlen field and the actual\n        padding is subtracted to the string that was provided to the pre_dissect  # noqa: E501\n        fun of the pkt parameter\n        :return: int; length of the data part of the HTTP/2 frame packet provided as parameter  # noqa: E501\n        :raises: AssertionError\n        '
        padding_len = self.getfieldval('padlen')
        (fld, fval) = self.getfield_and_val('padlen')
        padding_len_len = fld.i2len(self, fval)
        ret = self.s_len - padding_len_len - padding_len
        assert ret >= 0
        return ret

    def pre_dissect(self, s):
        if False:
            return 10
        'pre_dissect is filling the s_len property of this instance. This\n        property is later used during the getfield call of the "data" field when  # noqa: E501\n        trying to evaluate the length of the StrLenField! This "trick" works\n        because the underlayer packet (H2Frame) is assumed to override the\n        "extract_padding" method and to only provide to this packet the data\n        necessary for this packet. Tricky, tricky, will break some day probably!  # noqa: E501\n        '
        self.s_len = len(s)
        return s

class H2AbstractHeadersFrame(H2FramePayload):
    """Superclass of all variants of HTTP/2 Header Frame Packets.
    May be used for type checking.
    """

class H2HeadersFrame(H2AbstractHeadersFrame):
    """ H2HeadersFrame implements RFC 7540 par6.2 Headers Frame
    when there is no padding and no priority information

    The choice of decomposing into four classes is probably preferable to having  # noqa: E501
    numerous conditional fields based on the underlayer :/
    """
    type_id = 1
    END_STREAM_FLAG = 0
    END_HEADERS_FLAG = 2
    PADDED_FLAG = 3
    PRIORITY_FLAG = 5
    flags = {END_STREAM_FLAG: fields.MultiFlagsEntry('ES', 'End Stream'), END_HEADERS_FLAG: fields.MultiFlagsEntry('EH', 'End Headers'), PADDED_FLAG: fields.MultiFlagsEntry('P', 'Padded'), PRIORITY_FLAG: fields.MultiFlagsEntry('+', 'Priority')}
    name = 'HTTP/2 Headers Frame'
    fields_desc = [fields.PacketListField('hdrs', [], HPackHeaders)]

class H2PaddedHeadersFrame(H2AbstractHeadersFrame):
    """ H2PaddedHeadersFrame is the variant of H2HeadersFrame where padding flag
    is set and priority flag is cleared
    """
    __slots__ = ['s_len']
    name = 'HTTP/2 Headers Frame with Padding'
    fields_desc = [fields.FieldLenField('padlen', None, length_of='padding', fmt='B'), fields.PacketListField('hdrs', [], HPackHeaders, length_from=lambda pkt: pkt.get_hdrs_len()), fields.StrLenField('padding', '', length_from=lambda pkt: pkt.getfieldval('padlen'))]

    def get_hdrs_len(self):
        if False:
            while True:
                i = 10
        ' get_hdrs_len computes the length of the hdrs field\n\n        To do this computation, the length of the padlen field and the actual\n        padding is subtracted to the string that was provided to the pre_dissect  # noqa: E501\n        fun of the pkt parameter.\n        :return: int; length of the data part of the HTTP/2 frame packet provided as parameter  # noqa: E501\n        :raises: AssertionError\n        '
        padding_len = self.getfieldval('padlen')
        (fld, fval) = self.getfield_and_val('padlen')
        padding_len_len = fld.i2len(self, fval)
        ret = self.s_len - padding_len_len - padding_len
        assert ret >= 0
        return ret

    def pre_dissect(self, s):
        if False:
            for i in range(10):
                print('nop')
        'pre_dissect is filling the s_len property of this instance. This\n        property is later used during the parsing of the hdrs PacketListField\n        when trying to evaluate the length of the PacketListField! This "trick"\n        works because the underlayer packet (H2Frame) is assumed to override the  # noqa: E501\n        "extract_padding" method and to only provide to this packet the data\n        necessary for this packet. Tricky, tricky, will break some day probably!  # noqa: E501\n        '
        self.s_len = len(s)
        return s

class H2PriorityHeadersFrame(H2AbstractHeadersFrame):
    """ H2PriorityHeadersFrame is the variant of H2HeadersFrame where priority flag
    is set and padding flag is cleared
    """
    __slots__ = ['s_len']
    name = 'HTTP/2 Headers Frame with Priority'
    fields_desc = [fields.BitField('exclusive', 0, 1), fields.BitField('stream_dependency', 0, 31), fields.ByteField('weight', 0), fields.PacketListField('hdrs', [], HPackHeaders)]

class H2PaddedPriorityHeadersFrame(H2AbstractHeadersFrame):
    """ H2PaddedPriorityHeadersFrame is the variant of H2HeadersFrame where
    both priority and padding flags are set
    """
    __slots__ = ['s_len']
    name = 'HTTP/2 Headers Frame with Padding and Priority'
    fields_desc = [fields.FieldLenField('padlen', None, length_of='padding', fmt='B'), fields.BitField('exclusive', 0, 1), fields.BitField('stream_dependency', 0, 31), fields.ByteField('weight', 0), fields.PacketListField('hdrs', [], HPackHeaders, length_from=lambda pkt: pkt.get_hdrs_len()), fields.StrLenField('padding', '', length_from=lambda pkt: pkt.getfieldval('padlen'))]

    def get_hdrs_len(self):
        if False:
            return 10
        ' get_hdrs_len computes the length of the hdrs field\n\n        To do this computation, the length of the padlen field, the priority\n        information fields and the actual padding is subtracted to the string\n        that was provided to the pre_dissect fun of the pkt parameter.\n        :return: int: the length of the hdrs field\n        :raises: AssertionError\n        '
        padding_len = self.getfieldval('padlen')
        (fld, fval) = self.getfield_and_val('padlen')
        padding_len_len = fld.i2len(self, fval)
        bit_cnt = self.get_field('exclusive').size
        bit_cnt += self.get_field('stream_dependency').size
        (fld, fval) = self.getfield_and_val('weight')
        weight_len = fld.i2len(self, fval)
        ret = int(self.s_len - padding_len_len - padding_len - bit_cnt / 8 - weight_len)
        assert ret >= 0
        return ret

    def pre_dissect(self, s):
        if False:
            return 10
        'pre_dissect is filling the s_len property of this instance. This\n        property is later used during the parsing of the hdrs PacketListField\n        when trying to evaluate the length of the PacketListField! This "trick"\n        works because the underlayer packet (H2Frame) is assumed to override the  # noqa: E501\n        "extract_padding" method and to only provide to this packet the data\n        necessary for this packet. Tricky, tricky, will break some day probably!  # noqa: E501\n        '
        self.s_len = len(s)
        return s

class H2PriorityFrame(H2FramePayload):
    """ H2PriorityFrame implements RFC 7540 par6.3
    """
    type_id = 2
    name = 'HTTP/2 Priority Frame'
    fields_desc = [fields.BitField('exclusive', 0, 1), fields.BitField('stream_dependency', 0, 31), fields.ByteField('weight', 0)]

class H2ErrorCodes(object):
    """ H2ErrorCodes is an enumeration of the error codes defined in
    RFC7540 par7.
    This enumeration is not part of any frame because the error codes are in
    common with H2ResetFrame and H2GoAwayFrame.
    """
    NO_ERROR = 0
    PROTOCOL_ERROR = 1
    INTERNAL_ERROR = 2
    FLOW_CONTROL_ERROR = 3
    SETTINGS_TIMEOUT = 4
    STREAM_CLOSED = 5
    FRAME_SIZE_ERROR = 6
    REFUSED_STREAM = 7
    CANCEL = 8
    COMPRESSION_ERROR = 9
    CONNECT_ERROR = 10
    ENHANCE_YOUR_CALM = 11
    INADEQUATE_SECURITY = 12
    HTTP_1_1_REQUIRED = 13
    literal = {NO_ERROR: 'No error', PROTOCOL_ERROR: 'Protocol error', INTERNAL_ERROR: 'Internal error', FLOW_CONTROL_ERROR: 'Flow control error', SETTINGS_TIMEOUT: 'Settings timeout', STREAM_CLOSED: 'Stream closed', FRAME_SIZE_ERROR: 'Frame size error', REFUSED_STREAM: 'Refused stream', CANCEL: 'Cancel', COMPRESSION_ERROR: 'Compression error', CONNECT_ERROR: 'Control error', ENHANCE_YOUR_CALM: 'Enhance your calm', INADEQUATE_SECURITY: 'Inadequate security', HTTP_1_1_REQUIRED: 'HTTP/1.1 required'}

class H2ResetFrame(H2FramePayload):
    """ H2ResetFrame implements RFC 7540 par6.4
    """
    type_id = 3
    name = 'HTTP/2 Reset Frame'
    fields_desc = [fields.EnumField('error', 0, H2ErrorCodes.literal, fmt='!I')]

class H2Setting(packet.Packet):
    """ H2Setting implements a setting, as defined in RFC7540 par6.5.1
    """
    SETTINGS_HEADER_TABLE_SIZE = 1
    SETTINGS_ENABLE_PUSH = 2
    SETTINGS_MAX_CONCURRENT_STREAMS = 3
    SETTINGS_INITIAL_WINDOW_SIZE = 4
    SETTINGS_MAX_FRAME_SIZE = 5
    SETTINGS_MAX_HEADER_LIST_SIZE = 6
    name = 'HTTP/2 Setting'
    fields_desc = [fields.EnumField('id', 0, {SETTINGS_HEADER_TABLE_SIZE: 'Header table size', SETTINGS_ENABLE_PUSH: 'Enable push', SETTINGS_MAX_CONCURRENT_STREAMS: 'Max concurrent streams', SETTINGS_INITIAL_WINDOW_SIZE: 'Initial window size', SETTINGS_MAX_FRAME_SIZE: 'Max frame size', SETTINGS_MAX_HEADER_LIST_SIZE: 'Max header list size'}, fmt='!H'), fields.IntField('value', 0)]

    def guess_payload_class(self, payload):
        if False:
            i = 10
            return i + 15
        return config.conf.padding_layer

class H2SettingsFrame(H2FramePayload):
    """ H2SettingsFrame implements RFC7540 par6.5
    """
    type_id = 4
    ACK_FLAG = 0
    flags = {ACK_FLAG: fields.MultiFlagsEntry('A', 'ACK')}
    name = 'HTTP/2 Settings Frame'
    fields_desc = [fields.PacketListField('settings', [], H2Setting)]

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '__init__ initializes this H2SettingsFrame\n\n        If a _pkt arg is provided (by keyword), then this is an initialization\n        from a string to dissect and therefore the length of the string to\n        dissect have distinctive characteristics that we might want to check.\n        This is possible because the underlayer packet (H2Frame) overrides\n        extract_padding method to provided only the string that must be parsed\n        by this packet!\n        :raises: AssertionError\n        '
        assert len(args) == 0 or (isinstance(args[0], bytes) and len(args[0]) % 6 == 0), 'Invalid settings frame; length is not a multiple of 6'
        super(H2SettingsFrame, self).__init__(*args, **kwargs)

class H2PushPromiseFrame(H2FramePayload):
    """ H2PushPromiseFrame implements RFC7540 par6.6. This packet
    is the variant to use when the underlayer padding flag is cleared
    """
    type_id = 5
    END_HEADERS_FLAG = 2
    PADDED_FLAG = 3
    flags = {END_HEADERS_FLAG: fields.MultiFlagsEntry('EH', 'End Headers'), PADDED_FLAG: fields.MultiFlagsEntry('P', 'Padded')}
    name = 'HTTP/2 Push Promise Frame'
    fields_desc = [fields.BitField('reserved', 0, 1), fields.BitField('stream_id', 0, 31), fields.PacketListField('hdrs', [], HPackHeaders)]

class H2PaddedPushPromiseFrame(H2PushPromiseFrame):
    """ H2PaddedPushPromiseFrame implements RFC7540 par6.6. This
    packet is the variant to use when the underlayer padding flag is set
    """
    __slots__ = ['s_len']
    name = 'HTTP/2 Padded Push Promise Frame'
    fields_desc = [fields.FieldLenField('padlen', None, length_of='padding', fmt='B'), fields.BitField('reserved', 0, 1), fields.BitField('stream_id', 0, 31), fields.PacketListField('hdrs', [], HPackHeaders, length_from=lambda pkt: pkt.get_hdrs_len()), fields.StrLenField('padding', '', length_from=lambda pkt: pkt.getfieldval('padlen'))]

    def get_hdrs_len(self):
        if False:
            for i in range(10):
                print('nop')
        ' get_hdrs_len computes the length of the hdrs field\n\n        To do this computation, the length of the padlen field, reserved,\n        stream_id and the actual padding is subtracted to the string that was\n        provided to the pre_dissect fun of the pkt parameter.\n        :return: int: the length of the hdrs field\n        :raises: AssertionError\n        '
        (fld, padding_len) = self.getfield_and_val('padlen')
        padding_len_len = fld.i2len(self, padding_len)
        bit_len = self.get_field('reserved').size
        bit_len += self.get_field('stream_id').size
        ret = int(self.s_len - padding_len_len - padding_len - bit_len / 8)
        assert ret >= 0
        return ret

    def pre_dissect(self, s):
        if False:
            return 10
        'pre_dissect is filling the s_len property of this instance. This\n        property is later used during the parsing of the hdrs PacketListField\n        when trying to evaluate the length of the PacketListField! This "trick"\n        works because the underlayer packet (H2Frame) is assumed to override the  # noqa: E501\n        "extract_padding" method and to only provide to this packet the data\n        necessary for this packet. Tricky, tricky, will break some day probably!  # noqa: E501\n        '
        self.s_len = len(s)
        return s

class H2PingFrame(H2FramePayload):
    """ H2PingFrame implements the RFC 7540 par6.7
    """
    type_id = 6
    ACK_FLAG = 0
    flags = {ACK_FLAG: fields.MultiFlagsEntry('A', 'ACK')}
    name = 'HTTP/2 Ping Frame'
    fields_desc = [fields.LongField('opaque', 0)]

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        :raises: AssertionError\n        '
        assert len(args) == 0 or (isinstance(args[0], (bytes, str)) and len(args[0]) == 8), 'Invalid ping frame; length is not 8'
        super(H2PingFrame, self).__init__(*args, **kwargs)

class H2GoAwayFrame(H2FramePayload):
    """ H2GoAwayFrame implements the RFC 7540 par6.8
    """
    type_id = 7
    name = 'HTTP/2 Go Away Frame'
    fields_desc = [fields.BitField('reserved', 0, 1), fields.BitField('last_stream_id', 0, 31), fields.EnumField('error', 0, H2ErrorCodes.literal, fmt='!I'), fields.StrField('additional_data', '')]

class H2WindowUpdateFrame(H2FramePayload):
    """ H2WindowUpdateFrame implements the RFC 7540 par6.9
    """
    type_id = 8
    name = 'HTTP/2 Window Update Frame'
    fields_desc = [fields.BitField('reserved', 0, 1), fields.BitField('win_size_incr', 0, 31)]

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        :raises: AssertionError\n        '
        assert len(args) == 0 or (isinstance(args[0], (bytes, str)) and len(args[0]) == 4), 'Invalid window update frame; length is not 4'
        super(H2WindowUpdateFrame, self).__init__(*args, **kwargs)

class H2ContinuationFrame(H2FramePayload):
    """ H2ContinuationFrame implements the RFC 7540 par6.10
    """
    type_id = 9
    END_HEADERS_FLAG = 2
    flags = {END_HEADERS_FLAG: fields.MultiFlagsEntry('EH', 'End Headers')}
    name = 'HTTP/2 Continuation Frame'
    fields_desc = [fields.PacketListField('hdrs', [], HPackHeaders)]
_HTTP2_types = {0: 'DataFrm', 1: 'HdrsFrm', 2: 'PrioFrm', 3: 'RstFrm', 4: 'SetFrm', 5: 'PushFrm', 6: 'PingFrm', 7: 'GoawayFrm', 8: 'WinFrm', 9: 'ContFrm'}

class H2Frame(packet.Packet):
    """ H2Frame implements the frame structure as defined in RFC 7540 par4.1

    This packet may have a payload (one of the H2FramePayload) or none, in some
    rare cases such as settings acknowledgement)
    """
    name = 'HTTP/2 Frame'
    fields_desc = [fields.X3BytesField('len', None), fields.EnumField('type', None, _HTTP2_types, 'b'), fields.MultiFlagsField('flags', set(), 8, {H2DataFrame.type_id: H2DataFrame.flags, H2HeadersFrame.type_id: H2HeadersFrame.flags, H2PushPromiseFrame.type_id: H2PushPromiseFrame.flags, H2SettingsFrame.type_id: H2SettingsFrame.flags, H2PingFrame.type_id: H2PingFrame.flags, H2ContinuationFrame.type_id: H2ContinuationFrame.flags}, depends_on=lambda pkt: pkt.getfieldval('type')), fields.BitField('reserved', 0, 1), fields.BitField('stream_id', 0, 31)]

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        ' guess_payload_class returns the Class object to use for parsing a payload\n        This function uses the H2Frame.type field value to decide which payload to parse. The implement cannot be  # noqa: E501\n        performed using the simple bind_layers helper because sometimes the selection of which Class object to return  # noqa: E501\n        also depends on the H2Frame.flags value.\n\n        :param payload:\n        :return::\n        '
        if len(payload) == 0:
            return packet.NoPayload
        t = self.getfieldval('type')
        if t == H2DataFrame.type_id:
            if H2DataFrame.flags[H2DataFrame.PADDED_FLAG].short in self.getfieldval('flags'):
                return H2PaddedDataFrame
            return H2DataFrame
        if t == H2HeadersFrame.type_id:
            if H2HeadersFrame.flags[H2HeadersFrame.PADDED_FLAG].short in self.getfieldval('flags'):
                if H2HeadersFrame.flags[H2HeadersFrame.PRIORITY_FLAG].short in self.getfieldval('flags'):
                    return H2PaddedPriorityHeadersFrame
                else:
                    return H2PaddedHeadersFrame
            elif H2HeadersFrame.flags[H2HeadersFrame.PRIORITY_FLAG].short in self.getfieldval('flags'):
                return H2PriorityHeadersFrame
            return H2HeadersFrame
        if t == H2PriorityFrame.type_id:
            return H2PriorityFrame
        if t == H2ResetFrame.type_id:
            return H2ResetFrame
        if t == H2SettingsFrame.type_id:
            return H2SettingsFrame
        if t == H2PushPromiseFrame.type_id:
            if H2PushPromiseFrame.flags[H2PushPromiseFrame.PADDED_FLAG].short in self.getfieldval('flags'):
                return H2PaddedPushPromiseFrame
            return H2PushPromiseFrame
        if t == H2PingFrame.type_id:
            return H2PingFrame
        if t == H2GoAwayFrame.type_id:
            return H2GoAwayFrame
        if t == H2WindowUpdateFrame.type_id:
            return H2WindowUpdateFrame
        if t == H2ContinuationFrame.type_id:
            return H2ContinuationFrame
        return config.conf.padding_layer

    def extract_padding(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param str s: the string from which to tell the padding and the payload data apart  # noqa: E501\n        :return: (str, str): the padding and the payload data strings\n        :raises: AssertionError\n        '
        assert isinstance(self.len, int) and self.len >= 0, 'Invalid length: negative len?'
        assert len(s) >= self.len, 'Invalid length: string too short for this length'
        return (s[:self.len], s[self.len:])

    def post_build(self, p, pay):
        if False:
            while True:
                i = 10
        '\n        :param str p: the stringified packet\n        :param str pay: the stringified payload\n        :return: str: the stringified packet and payload, with the packet length field "patched"  # noqa: E501\n        :raises: AssertionError\n        '
        if self.getfieldval('len') is None:
            assert len(pay) < 1 << 24, 'Invalid length: payload is too long'
            p = struct.pack('!L', len(pay))[1:] + p[3:]
        return super(H2Frame, self).post_build(p, pay)

class H2Seq(packet.Packet):
    """ H2Seq is a helper packet that contains several H2Frames and their
    payload. This packet can be used, for instance, while reading manually from
    a TCP socket.
    """
    name = 'HTTP/2 Frame Sequence'
    fields_desc = [fields.PacketListField('frames', [], H2Frame)]

    def guess_payload_class(self, payload):
        if False:
            print('Hello World!')
        return config.conf.padding_layer
packet.bind_layers(H2Frame, H2DataFrame, {'type': H2DataFrame.type_id})
packet.bind_layers(H2Frame, H2PaddedDataFrame, {'type': H2DataFrame.type_id})
packet.bind_layers(H2Frame, H2HeadersFrame, {'type': H2HeadersFrame.type_id})
packet.bind_layers(H2Frame, H2PaddedHeadersFrame, {'type': H2HeadersFrame.type_id})
packet.bind_layers(H2Frame, H2PriorityHeadersFrame, {'type': H2HeadersFrame.type_id})
packet.bind_layers(H2Frame, H2PaddedPriorityHeadersFrame, {'type': H2HeadersFrame.type_id})
packet.bind_layers(H2Frame, H2PriorityFrame, {'type': H2PriorityFrame.type_id})
packet.bind_layers(H2Frame, H2ResetFrame, {'type': H2ResetFrame.type_id})
packet.bind_layers(H2Frame, H2SettingsFrame, {'type': H2SettingsFrame.type_id})
packet.bind_layers(H2Frame, H2PingFrame, {'type': H2PingFrame.type_id})
packet.bind_layers(H2Frame, H2PushPromiseFrame, {'type': H2PushPromiseFrame.type_id})
packet.bind_layers(H2Frame, H2PaddedPushPromiseFrame, {'type': H2PaddedPushPromiseFrame.type_id})
packet.bind_layers(H2Frame, H2GoAwayFrame, {'type': H2GoAwayFrame.type_id})
packet.bind_layers(H2Frame, H2WindowUpdateFrame, {'type': H2WindowUpdateFrame.type_id})
packet.bind_layers(H2Frame, H2ContinuationFrame, {'type': H2ContinuationFrame.type_id})
H2_CLIENT_CONNECTION_PREFACE = hex_bytes('505249202a20485454502f322e300d0a0d0a534d0d0a0d0a')

class HPackHdrEntry(Sized):
    """ HPackHdrEntry is an entry of the HPackHdrTable helper

    Each HPackHdrEntry instance is a header line (name and value). Names are
    normalized (lowercase), according to RFC 7540 par8.1.2
    """
    __slots__ = ['_name', '_len', '_value']

    def __init__(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        :raises: AssertionError\n        '
        assert len(name) > 0
        self._name = name.lower()
        self._value = value
        self._len = 32 + len(self._name) + len(self._value)

    def name(self):
        if False:
            i = 10
            return i + 15
        return self._name

    def value(self):
        if False:
            i = 10
            return i + 15
        return self._value

    def size(self):
        if False:
            i = 10
            return i + 15
        ' size returns the "length" of the header entry, as defined in\n        RFC 7541 par4.1.\n        '
        return self._len
    __len__ = size

    def __str__(self):
        if False:
            print('Hello World!')
        ' __str__ returns the header as it would be formatted in textual format\n        '
        if self._name.startswith(':'):
            return '{} {}'.format(self._name, self._value)
        else:
            return '{}: {}'.format(self._name, self._value)

    def __bytes__(self):
        if False:
            return 10
        return bytes_encode(self.__str__())

class HPackHdrTable(Sized):
    """ HPackHdrTable is a helper class that implements some of the logic
    associated with indexing of headers (read and write operations in this
    "registry". THe HPackHdrTable also implements convenience functions to easily  # noqa: E501
    convert to and from textual representation and binary representation of
    a HTTP/2 requests
    """
    __slots__ = ['_dynamic_table', '_dynamic_table_max_size', '_dynamic_table_cap_size', '_regexp']
    '\n    :var _dynamic_table: the list containing entries requested to be added by\n        the peer and registered with a register() call\n    :var _dynamic_table_max_size: the current maximum size of the dynamic table\n        in bytes. This value is updated with the Dynamic Table Size Update\n        messages defined in RFC 7541 par6.3\n    :var _dynamic_table_cap_size: the maximum size of the dynamic table in\n        bytes. This value is updated with the SETTINGS_HEADER_TABLE_SIZE HTTP/2\n        setting.\n    '
    _static_entries = {1: HPackHdrEntry(':authority', ''), 2: HPackHdrEntry(':method', 'GET'), 3: HPackHdrEntry(':method', 'POST'), 4: HPackHdrEntry(':path', '/'), 5: HPackHdrEntry(':path', '/index.html'), 6: HPackHdrEntry(':scheme', 'http'), 7: HPackHdrEntry(':scheme', 'https'), 8: HPackHdrEntry(':status', '200'), 9: HPackHdrEntry(':status', '204'), 10: HPackHdrEntry(':status', '206'), 11: HPackHdrEntry(':status', '304'), 12: HPackHdrEntry(':status', '400'), 13: HPackHdrEntry(':status', '404'), 14: HPackHdrEntry(':status', '500'), 15: HPackHdrEntry('accept-charset', ''), 16: HPackHdrEntry('accept-encoding', 'gzip, deflate'), 17: HPackHdrEntry('accept-language', ''), 18: HPackHdrEntry('accept-ranges', ''), 19: HPackHdrEntry('accept', ''), 20: HPackHdrEntry('access-control-allow-origin', ''), 21: HPackHdrEntry('age', ''), 22: HPackHdrEntry('allow', ''), 23: HPackHdrEntry('authorization', ''), 24: HPackHdrEntry('cache-control', ''), 25: HPackHdrEntry('content-disposition', ''), 26: HPackHdrEntry('content-encoding', ''), 27: HPackHdrEntry('content-language', ''), 28: HPackHdrEntry('content-length', ''), 29: HPackHdrEntry('content-location', ''), 30: HPackHdrEntry('content-range', ''), 31: HPackHdrEntry('content-type', ''), 32: HPackHdrEntry('cookie', ''), 33: HPackHdrEntry('date', ''), 34: HPackHdrEntry('etag', ''), 35: HPackHdrEntry('expect', ''), 36: HPackHdrEntry('expires', ''), 37: HPackHdrEntry('from', ''), 38: HPackHdrEntry('host', ''), 39: HPackHdrEntry('if-match', ''), 40: HPackHdrEntry('if-modified-since', ''), 41: HPackHdrEntry('if-none-match', ''), 42: HPackHdrEntry('if-range', ''), 43: HPackHdrEntry('if-unmodified-since', ''), 44: HPackHdrEntry('last-modified', ''), 45: HPackHdrEntry('link', ''), 46: HPackHdrEntry('location', ''), 47: HPackHdrEntry('max-forwards', ''), 48: HPackHdrEntry('proxy-authenticate', ''), 49: HPackHdrEntry('proxy-authorization', ''), 50: HPackHdrEntry('range', ''), 51: HPackHdrEntry('referer', ''), 52: HPackHdrEntry('refresh', ''), 53: HPackHdrEntry('retry-after', ''), 54: HPackHdrEntry('server', ''), 55: HPackHdrEntry('set-cookie', ''), 56: HPackHdrEntry('strict-transport-security', ''), 57: HPackHdrEntry('transfer-encoding', ''), 58: HPackHdrEntry('user-agent', ''), 59: HPackHdrEntry('vary', ''), 60: HPackHdrEntry('via', ''), 61: HPackHdrEntry('www-authenticate', '')}
    _static_entries_last_idx = None

    @classmethod
    def init_static_table(cls):
        if False:
            return 10
        cls._static_entries_last_idx = max(cls._static_entries)

    def __init__(self, dynamic_table_max_size=4096, dynamic_table_cap_size=4096):
        if False:
            print('Hello World!')
        '\n        :param int dynamic_table_max_size: the current maximum size of the dynamic entry table in bytes  # noqa: E501\n        :param int dynamic_table_cap_size: the maximum-maximum size of the dynamic entry table in bytes  # noqa: E501\n        :raises:s AssertionError\n        '
        self._regexp = None
        if isinstance(type(self)._static_entries_last_idx, type(None)):
            type(self).init_static_table()
        assert dynamic_table_max_size <= dynamic_table_cap_size, 'EINVAL: dynamic_table_max_size too large; expected value is less or equal to dynamic_table_cap_size'
        self._dynamic_table = []
        self._dynamic_table_max_size = dynamic_table_max_size
        self._dynamic_table_cap_size = dynamic_table_cap_size

    def __getitem__(self, idx):
        if False:
            print('Hello World!')
        'Gets an element from the header tables (static or dynamic indifferently)\n\n        :param int idx: the index number of the entry to retrieve. If the index\n        value is superior to the last index of the static entry table, then the\n        dynamic entry type is requested, following the procedure described in\n        RFC 7541 par2.3.3\n        :return: HPackHdrEntry: the entry defined at this requested index. If the entry does not exist, KeyError is  # noqa: E501\n          raised\n        :raises: KeyError, AssertionError\n        '
        assert idx >= 0
        if idx > type(self)._static_entries_last_idx:
            idx -= type(self)._static_entries_last_idx + 1
            if idx >= len(self._dynamic_table):
                raise KeyError('EINVAL: idx: out-of-bound read: {}; maximum index: {}'.format(idx, len(self._dynamic_table)))
            return self._dynamic_table[idx]
        return type(self)._static_entries[idx]

    def resize(self, ns):
        if False:
            print('Hello World!')
        'Resize the dynamic table. If the new size (ns) must be between 0 and\n        the cap size. If the new size is lower than the current size of the\n        dynamic table, entries are evicted.\n        :param int ns: the new size of the dynamic table\n        :raises: AssertionError\n        '
        assert 0 <= ns <= self._dynamic_table_cap_size, 'EINVAL: ns: out-of-range value; expected value is in the range [0;{}['.format(self._dynamic_table_cap_size)
        old_size = self._dynamic_table_max_size
        self._dynamic_table_max_size = ns
        if old_size > self._dynamic_table_max_size:
            self._reduce_dynamic_table()

    def recap(self, nc):
        if False:
            i = 10
            return i + 15
        'recap changes the maximum size limit of the dynamic table. It also\n        proceeds to a resize(), if the new size is lower than the previous one.\n        :param int nc: the new cap of the dynamic table (that is the maximum-maximum size)  # noqa: E501\n        :raises: AssertionError\n        '
        assert nc >= 0
        t = self._dynamic_table_cap_size > nc
        self._dynamic_table_cap_size = nc
        if t:
            self.resize(nc)

    def _reduce_dynamic_table(self, new_entry_size=0):
        if False:
            print('Hello World!')
        '_reduce_dynamic_table evicts entries from the dynamic table until it\n        fits in less than the current size limit. The optional parameter,\n        new_entry_size, allows the resize to happen so that a new entry of this\n        size fits in.\n        :param int new_entry_size: if called before adding a new entry, the size of the new entry in bytes (following  # noqa: E501\n        the RFC7541 definition of the size of an entry)\n        :raises: AssertionError\n        '
        assert new_entry_size >= 0
        cur_sz = len(self)
        dyn_tbl_sz = len(self._dynamic_table)
        while dyn_tbl_sz > 0 and cur_sz + new_entry_size > self._dynamic_table_max_size:
            last_elmt_sz = len(self._dynamic_table[-1])
            self._dynamic_table.pop()
            dyn_tbl_sz -= 1
            cur_sz -= last_elmt_sz

    def register(self, hdrs):
        if False:
            return 10
        'register adds to this table the instances of\n        HPackLitHdrFldWithIncrIndexing provided as parameters.\n\n        A H2Frame with a H2HeadersFrame payload can be provided, as much as a\n        python list of HPackHeaders or a single HPackLitHdrFldWithIncrIndexing\n        instance.\n        :param HPackLitHdrFldWithIncrIndexing|H2Frame|list of HPackHeaders hdrs: the header(s) to register  # noqa: E501\n        :raises: AssertionError\n        '
        if isinstance(hdrs, H2Frame):
            hdrs = [hdr for hdr in hdrs.payload.hdrs if isinstance(hdr, HPackLitHdrFldWithIncrIndexing)]
        elif isinstance(hdrs, HPackLitHdrFldWithIncrIndexing):
            hdrs = [hdrs]
        else:
            hdrs = [hdr for hdr in hdrs if isinstance(hdr, HPackLitHdrFldWithIncrIndexing)]
        for hdr in hdrs:
            if hdr.index == 0:
                hdr_name = hdr.hdr_name.getfieldval('data').origin()
            else:
                idx = int(hdr.index)
                hdr_name = self[idx].name()
            hdr_value = hdr.hdr_value.getfieldval('data').origin()
            entry = HPackHdrEntry(hdr_name, hdr_value)
            new_entry_len = len(entry)
            self._reduce_dynamic_table(new_entry_len)
            assert new_entry_len <= self._dynamic_table_max_size
            self._dynamic_table.insert(0, entry)

    def get_idx_by_name(self, name):
        if False:
            while True:
                i = 10
        ' get_idx_by_name returns the index of a matching registered header\n\n        This implementation will prefer returning a static entry index whenever\n        possible. If multiple matching header name are found in the static\n        table, there is insurance that the first entry (lowest index number)\n        will be returned.\n        If no matching header is found, this method returns None.\n        '
        name = name.lower()
        for (key, val) in type(self)._static_entries.items():
            if val.name() == name:
                return key
        for (idx, val) in enumerate(self._dynamic_table):
            if val.name() == name:
                return type(self)._static_entries_last_idx + idx + 1
        return None

    def get_idx_by_name_and_value(self, name, value):
        if False:
            i = 10
            return i + 15
        ' get_idx_by_name_and_value returns the index of a matching registered\n        header\n\n        This implementation will prefer returning a static entry index whenever\n        possible. If multiple matching headers are found in the dynamic table,\n        the lowest index is returned\n        If no matching header is found, this method returns None.\n        '
        name = name.lower()
        for (key, val) in type(self)._static_entries.items():
            if val.name() == name and val.value() == value:
                return key
        for (idx, val) in enumerate(self._dynamic_table):
            if val.name() == name and val.value() == value:
                return type(self)._static_entries_last_idx + idx + 1
        return None

    def __len__(self):
        if False:
            return 10
        ' __len__ returns the summed length of all dynamic entries\n        '
        return sum((len(x) for x in self._dynamic_table))

    def gen_txt_repr(self, hdrs, register=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        gen_txt_repr returns a "textual" representation of the provided\n        headers.\n        The output of this function is compatible with the input of\n        parse_txt_hdrs.\n\n        :param H2Frame|list of HPackHeaders hdrs: the list of headers to\n          convert to textual representation.\n        :param bool: whether incremental headers should be added to the dynamic\n          table as we generate the text representation\n        :return: str: the textual representation of the provided headers\n        :raises: AssertionError\n        '
        lst = []
        if isinstance(hdrs, H2Frame):
            hdrs = hdrs.payload.hdrs
        for hdr in hdrs:
            try:
                if isinstance(hdr, HPackIndexedHdr):
                    lst.append('{}'.format(self[hdr.index]))
                elif isinstance(hdr, (HPackLitHdrFldWithIncrIndexing, HPackLitHdrFldWithoutIndexing)):
                    if hdr.index != 0:
                        name = self[hdr.index].name()
                    else:
                        name = hdr.hdr_name.getfieldval('data').origin()
                    if name.startswith(':'):
                        lst.append('{} {}'.format(name, hdr.hdr_value.getfieldval('data').origin()))
                    else:
                        lst.append('{}: {}'.format(name, hdr.hdr_value.getfieldval('data').origin()))
                if register and isinstance(hdr, HPackLitHdrFldWithIncrIndexing):
                    self.register(hdr)
            except KeyError as e:
                print(e)
                continue
        return '\n'.join(lst)

    @staticmethod
    def _optimize_header_length_and_packetify(s):
        if False:
            i = 10
            return i + 15
        zs = HPackZString(s)
        if len(zs) >= len(s):
            return HPackHdrString(data=HPackLiteralString(s))
        return HPackHdrString(data=zs)

    def _convert_a_header_to_a_h2_header(self, hdr_name, hdr_value, is_sensitive, should_index):
        if False:
            i = 10
            return i + 15
        ' _convert_a_header_to_a_h2_header builds a HPackHeaders from a header\n        name and a value. It returns a HPackIndexedHdr whenever possible. If not,  # noqa: E501\n        it returns a HPackLitHdrFldWithoutIndexing or a\n        HPackLitHdrFldWithIncrIndexing, based on the should_index callback.\n        HPackLitHdrFldWithoutIndexing is forced if the is_sensitive callback\n        returns True and its never_index bit is set.\n        '
        idx = self.get_idx_by_name_and_value(hdr_name, hdr_value)
        if idx is not None:
            return (HPackIndexedHdr(index=idx), len(self[idx]))
        _hdr_value = self._optimize_header_length_and_packetify(hdr_value)
        idx = self.get_idx_by_name(hdr_name)
        if idx is not None:
            if is_sensitive(hdr_name, _hdr_value.getfieldval('data').origin()):
                return (HPackLitHdrFldWithoutIndexing(never_index=1, index=idx, hdr_value=_hdr_value), len(HPackHdrEntry(self[idx].name(), _hdr_value.getfieldval('data').origin())))
            if should_index(hdr_name):
                return (HPackLitHdrFldWithIncrIndexing(index=idx, hdr_value=_hdr_value), len(HPackHdrEntry(self[idx].name(), _hdr_value.getfieldval('data').origin())))
            return (HPackLitHdrFldWithoutIndexing(index=idx, hdr_value=_hdr_value), len(HPackHdrEntry(self[idx].name(), _hdr_value.getfieldval('data').origin())))
        _hdr_name = self._optimize_header_length_and_packetify(hdr_name)
        if is_sensitive(_hdr_name.getfieldval('data').origin(), _hdr_value.getfieldval('data').origin()):
            return (HPackLitHdrFldWithoutIndexing(never_index=1, index=0, hdr_name=_hdr_name, hdr_value=_hdr_value), len(HPackHdrEntry(_hdr_name.getfieldval('data').origin(), _hdr_value.getfieldval('data').origin())))
        if should_index(_hdr_name.getfieldval('data').origin()):
            return (HPackLitHdrFldWithIncrIndexing(index=0, hdr_name=_hdr_name, hdr_value=_hdr_value), len(HPackHdrEntry(_hdr_name.getfieldval('data').origin(), _hdr_value.getfieldval('data').origin())))
        return (HPackLitHdrFldWithoutIndexing(index=0, hdr_name=_hdr_name, hdr_value=_hdr_value), len(HPackHdrEntry(_hdr_name.getfieldval('data').origin(), _hdr_value.getfieldval('data').origin())))

    def _parse_header_line(self, line):
        if False:
            print('Hello World!')
        if self._regexp is None:
            self._regexp = re.compile(b'^(?::([a-z\\-0-9]+)|([a-z\\-0-9]+):)\\s+(.+)$')
        hdr_line = line.rstrip()
        grp = self._regexp.match(hdr_line)
        if grp is None or len(grp.groups()) != 3:
            return (None, None)
        if grp.group(1) is not None:
            hdr_name = b':' + grp.group(1)
        else:
            hdr_name = grp.group(2)
        return (plain_str(hdr_name.lower()), plain_str(grp.group(3)))

    def parse_txt_hdrs(self, s, stream_id=1, body=None, max_frm_sz=4096, max_hdr_lst_sz=0, is_sensitive=lambda n, v: False, should_index=lambda x: False, register=True):
        if False:
            i = 10
            return i + 15
        '\n        parse_txt_hdrs parses headers expressed in text and converts them\n        into a series of H2Frames with the "correct" flags. A body can be\n        provided in which case, the data frames are added, bearing the End\n        Stream flag, instead of the H2HeadersFrame/H2ContinuationFrame.\n        The generated frames may respect max_frm_sz (SETTINGS_MAX_FRAME_SIZE)\n        and max_hdr_lst_sz (SETTINGS_MAX_HEADER_LIST_SIZE) if provided.\n        The headers are split into multiple headers fragment (and H2Frames)\n        to respect these limits. Also, a callback can be provided to tell if\n        a header should be never indexed (sensitive headers, such as cookies),\n        and another callback say if the header should be registered into the\n        index table at all.\n        For an header to be registered, the is_sensitive callback must return\n        False AND the should_index callback should return True. This is the\n        default behavior.\n\n        :param str s: the string to parse for headers\n        :param int stream_id: the stream id to use in the generated H2Frames\n        :param str/None body: the eventual body of the request, that is added\n          to the generated frames\n        :param int max_frm_sz: the maximum frame size. This is used to split\n          the headers and data frames according to the maximum frame size\n          negotiated for this connection.\n        :param int max_hdr_lst_sz: the maximum size of a "header fragment" as\n          defined in RFC7540\n        :param callable is_sensitive: callback that returns True if the\n          provided header is sensible and must be stored in a header packet\n          requesting this header never to be indexed\n        :param callable should_index: callback that returns True if the\n          provided header should be stored in a header packet requesting\n          indexation in the dynamic header table.\n        :param bool register: whether to register new headers with incremental\n          indexing as we parse them\n        :raises: Exception\n        '
        sio = BytesIO(s.encode() if isinstance(s, str) else s)
        base_frm_len = len(raw(H2Frame()))
        ret = H2Seq()
        cur_frm = H2HeadersFrame()
        cur_hdr_sz = 0
        for hdr_line in sio:
            (hdr_name, hdr_value) = self._parse_header_line(hdr_line)
            if hdr_name is None:
                continue
            (new_hdr, new_hdr_len) = self._convert_a_header_to_a_h2_header(hdr_name, hdr_value, is_sensitive, should_index)
            new_hdr_bin_len = len(raw(new_hdr))
            if register and isinstance(new_hdr, HPackLitHdrFldWithIncrIndexing):
                self.register(new_hdr)
            if new_hdr_bin_len + base_frm_len > max_frm_sz or (max_hdr_lst_sz != 0 and new_hdr_len > max_hdr_lst_sz):
                raise Exception('Header too long: {}'.format(hdr_name))
            if max_frm_sz < len(raw(cur_frm)) + base_frm_len + new_hdr_len or (max_hdr_lst_sz != 0 and max_hdr_lst_sz < cur_hdr_sz + new_hdr_len):
                flags = set()
                if isinstance(cur_frm, H2HeadersFrame) and (not body):
                    flags.add('ES')
                ret.frames.append(H2Frame(stream_id=stream_id, flags=flags) / cur_frm)
                cur_frm = H2ContinuationFrame()
                cur_hdr_sz = 0
            hdr_list = cur_frm.hdrs
            hdr_list += new_hdr
            cur_hdr_sz += new_hdr_len
        flags = {'EH'}
        if isinstance(cur_frm, H2HeadersFrame) and (not body):
            flags.add('ES')
        ret.frames.append(H2Frame(stream_id=stream_id, flags=flags) / cur_frm)
        if body:
            base_data_frm_len = len(raw(H2DataFrame()))
            sio = BytesIO(body)
            frgmt = sio.read(max_frm_sz - base_data_frm_len - base_frm_len)
            while frgmt:
                nxt_frgmt = sio.read(max_frm_sz - base_data_frm_len - base_frm_len)
                flags = set()
                if len(nxt_frgmt) == 0:
                    flags.add('ES')
                ret.frames.append(H2Frame(stream_id=stream_id, flags=flags) / H2DataFrame(data=frgmt))
                frgmt = nxt_frgmt
        return ret