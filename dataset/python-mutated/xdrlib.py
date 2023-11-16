"""Implements (a subset of) Sun XDR -- eXternal Data Representation.

See: RFC 1014

"""
import struct
from io import BytesIO
from functools import wraps
__all__ = ['Error', 'Packer', 'Unpacker', 'ConversionError']

class Error(Exception):
    """Exception class for this module. Use:

    except xdrlib.Error as var:
        # var has the Error instance for the exception

    Public ivars:
        msg -- contains the message

    """

    def __init__(self, msg):
        if False:
            while True:
                i = 10
        self.msg = msg

    def __repr__(self):
        if False:
            return 10
        return repr(self.msg)

    def __str__(self):
        if False:
            return 10
        return str(self.msg)

class ConversionError(Error):
    pass

def raise_conversion_error(function):
    if False:
        return 10
    ' Wrap any raised struct.errors in a ConversionError. '

    @wraps(function)
    def result(self, value):
        if False:
            for i in range(10):
                print('nop')
        try:
            return function(self, value)
        except struct.error as e:
            raise ConversionError(e.args[0]) from None
    return result

class Packer:
    """Pack various data representations into a buffer."""

    def __init__(self):
        if False:
            return 10
        self.reset()

    def reset(self):
        if False:
            return 10
        self.__buf = BytesIO()

    def get_buffer(self):
        if False:
            return 10
        return self.__buf.getvalue()
    get_buf = get_buffer

    @raise_conversion_error
    def pack_uint(self, x):
        if False:
            return 10
        self.__buf.write(struct.pack('>L', x))

    @raise_conversion_error
    def pack_int(self, x):
        if False:
            i = 10
            return i + 15
        self.__buf.write(struct.pack('>l', x))
    pack_enum = pack_int

    def pack_bool(self, x):
        if False:
            while True:
                i = 10
        if x:
            self.__buf.write(b'\x00\x00\x00\x01')
        else:
            self.__buf.write(b'\x00\x00\x00\x00')

    def pack_uhyper(self, x):
        if False:
            i = 10
            return i + 15
        try:
            self.pack_uint(x >> 32 & 4294967295)
        except (TypeError, struct.error) as e:
            raise ConversionError(e.args[0]) from None
        try:
            self.pack_uint(x & 4294967295)
        except (TypeError, struct.error) as e:
            raise ConversionError(e.args[0]) from None
    pack_hyper = pack_uhyper

    @raise_conversion_error
    def pack_float(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.__buf.write(struct.pack('>f', x))

    @raise_conversion_error
    def pack_double(self, x):
        if False:
            return 10
        self.__buf.write(struct.pack('>d', x))

    def pack_fstring(self, n, s):
        if False:
            i = 10
            return i + 15
        if n < 0:
            raise ValueError('fstring size must be nonnegative')
        data = s[:n]
        n = (n + 3) // 4 * 4
        data = data + (n - len(data)) * b'\x00'
        self.__buf.write(data)
    pack_fopaque = pack_fstring

    def pack_string(self, s):
        if False:
            print('Hello World!')
        n = len(s)
        self.pack_uint(n)
        self.pack_fstring(n, s)
    pack_opaque = pack_string
    pack_bytes = pack_string

    def pack_list(self, list, pack_item):
        if False:
            print('Hello World!')
        for item in list:
            self.pack_uint(1)
            pack_item(item)
        self.pack_uint(0)

    def pack_farray(self, n, list, pack_item):
        if False:
            for i in range(10):
                print('nop')
        if len(list) != n:
            raise ValueError('wrong array size')
        for item in list:
            pack_item(item)

    def pack_array(self, list, pack_item):
        if False:
            print('Hello World!')
        n = len(list)
        self.pack_uint(n)
        self.pack_farray(n, list, pack_item)

class Unpacker:
    """Unpacks various data representations from the given buffer."""

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.reset(data)

    def reset(self, data):
        if False:
            i = 10
            return i + 15
        self.__buf = data
        self.__pos = 0

    def get_position(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__pos

    def set_position(self, position):
        if False:
            return 10
        self.__pos = position

    def get_buffer(self):
        if False:
            return 10
        return self.__buf

    def done(self):
        if False:
            i = 10
            return i + 15
        if self.__pos < len(self.__buf):
            raise Error('unextracted data remains')

    def unpack_uint(self):
        if False:
            while True:
                i = 10
        i = self.__pos
        self.__pos = j = i + 4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        return struct.unpack('>L', data)[0]

    def unpack_int(self):
        if False:
            print('Hello World!')
        i = self.__pos
        self.__pos = j = i + 4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        return struct.unpack('>l', data)[0]
    unpack_enum = unpack_int

    def unpack_bool(self):
        if False:
            return 10
        return bool(self.unpack_int())

    def unpack_uhyper(self):
        if False:
            while True:
                i = 10
        hi = self.unpack_uint()
        lo = self.unpack_uint()
        return int(hi) << 32 | lo

    def unpack_hyper(self):
        if False:
            return 10
        x = self.unpack_uhyper()
        if x >= 9223372036854775808:
            x = x - 18446744073709551616
        return x

    def unpack_float(self):
        if False:
            return 10
        i = self.__pos
        self.__pos = j = i + 4
        data = self.__buf[i:j]
        if len(data) < 4:
            raise EOFError
        return struct.unpack('>f', data)[0]

    def unpack_double(self):
        if False:
            while True:
                i = 10
        i = self.__pos
        self.__pos = j = i + 8
        data = self.__buf[i:j]
        if len(data) < 8:
            raise EOFError
        return struct.unpack('>d', data)[0]

    def unpack_fstring(self, n):
        if False:
            while True:
                i = 10
        if n < 0:
            raise ValueError('fstring size must be nonnegative')
        i = self.__pos
        j = i + (n + 3) // 4 * 4
        if j > len(self.__buf):
            raise EOFError
        self.__pos = j
        return self.__buf[i:i + n]
    unpack_fopaque = unpack_fstring

    def unpack_string(self):
        if False:
            return 10
        n = self.unpack_uint()
        return self.unpack_fstring(n)
    unpack_opaque = unpack_string
    unpack_bytes = unpack_string

    def unpack_list(self, unpack_item):
        if False:
            print('Hello World!')
        list = []
        while 1:
            x = self.unpack_uint()
            if x == 0:
                break
            if x != 1:
                raise ConversionError('0 or 1 expected, got %r' % (x,))
            item = unpack_item()
            list.append(item)
        return list

    def unpack_farray(self, n, unpack_item):
        if False:
            while True:
                i = 10
        list = []
        for i in range(n):
            list.append(unpack_item())
        return list

    def unpack_array(self, unpack_item):
        if False:
            while True:
                i = 10
        n = self.unpack_uint()
        return self.unpack_farray(n, unpack_item)