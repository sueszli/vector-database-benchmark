import struct
from io import BytesIO

class BCDataStream:

    def __init__(self, data=None):
        if False:
            i = 10
            return i + 15
        self.data = BytesIO(data)

    def reset(self):
        if False:
            while True:
                i = 10
        self.data.seek(0)

    def get_bytes(self):
        if False:
            print('Hello World!')
        return self.data.getvalue()

    def read(self, size):
        if False:
            print('Hello World!')
        return self.data.read(size)

    def write(self, data):
        if False:
            while True:
                i = 10
        self.data.write(data)

    def write_many(self, many):
        if False:
            i = 10
            return i + 15
        self.data.writelines(many)

    def read_string(self):
        if False:
            print('Hello World!')
        return self.read(self.read_compact_size())

    def write_string(self, s):
        if False:
            i = 10
            return i + 15
        self.write_compact_size(len(s))
        self.write(s)

    def read_compact_size(self):
        if False:
            for i in range(10):
                print('nop')
        size = self.read_uint8()
        if size < 253:
            return size
        if size == 253:
            return self.read_uint16()
        if size == 254:
            return self.read_uint32()
        if size == 255:
            return self.read_uint64()

    def write_compact_size(self, size):
        if False:
            while True:
                i = 10
        if size < 253:
            self.write_uint8(size)
        elif size <= 65535:
            self.write_uint8(253)
            self.write_uint16(size)
        elif size <= 4294967295:
            self.write_uint8(254)
            self.write_uint32(size)
        else:
            self.write_uint8(255)
            self.write_uint64(size)

    def read_boolean(self):
        if False:
            print('Hello World!')
        return self.read_uint8() != 0

    def write_boolean(self, val):
        if False:
            while True:
                i = 10
        return self.write_uint8(1 if val else 0)
    int8 = struct.Struct('b')
    uint8 = struct.Struct('B')
    int16 = struct.Struct('<h')
    uint16 = struct.Struct('<H')
    int32 = struct.Struct('<i')
    uint32 = struct.Struct('<I')
    int64 = struct.Struct('<q')
    uint64 = struct.Struct('<Q')

    def _read_struct(self, fmt):
        if False:
            while True:
                i = 10
        value = self.read(fmt.size)
        if value:
            return fmt.unpack(value)[0]

    def read_int8(self):
        if False:
            while True:
                i = 10
        return self._read_struct(self.int8)

    def read_uint8(self):
        if False:
            for i in range(10):
                print('nop')
        return self._read_struct(self.uint8)

    def read_int16(self):
        if False:
            for i in range(10):
                print('nop')
        return self._read_struct(self.int16)

    def read_uint16(self):
        if False:
            print('Hello World!')
        return self._read_struct(self.uint16)

    def read_int32(self):
        if False:
            for i in range(10):
                print('nop')
        return self._read_struct(self.int32)

    def read_uint32(self):
        if False:
            print('Hello World!')
        return self._read_struct(self.uint32)

    def read_int64(self):
        if False:
            return 10
        return self._read_struct(self.int64)

    def read_uint64(self):
        if False:
            for i in range(10):
                print('nop')
        return self._read_struct(self.uint64)

    def write_int8(self, val):
        if False:
            print('Hello World!')
        self.write(self.int8.pack(val))

    def write_uint8(self, val):
        if False:
            return 10
        self.write(self.uint8.pack(val))

    def write_int16(self, val):
        if False:
            return 10
        self.write(self.int16.pack(val))

    def write_uint16(self, val):
        if False:
            return 10
        self.write(self.uint16.pack(val))

    def write_int32(self, val):
        if False:
            while True:
                i = 10
        self.write(self.int32.pack(val))

    def write_uint32(self, val):
        if False:
            while True:
                i = 10
        self.write(self.uint32.pack(val))

    def write_int64(self, val):
        if False:
            while True:
                i = 10
        self.write(self.int64.pack(val))

    def write_uint64(self, val):
        if False:
            while True:
                i = 10
        self.write(self.uint64.pack(val))