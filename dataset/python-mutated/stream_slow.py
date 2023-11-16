import struct

class InputStream(object):
    """
    A pure Python implementation of InputStream.
    """

    def __init__(self, data):
        if False:
            return 10
        self.data = data
        self.pos = 0

    def read(self, size):
        if False:
            for i in range(10):
                print('nop')
        self.pos += size
        return self.data[self.pos - size:self.pos]

    def read_byte(self):
        if False:
            while True:
                i = 10
        self.pos += 1
        return self.data[self.pos - 1]

    def read_int8(self):
        if False:
            for i in range(10):
                print('nop')
        return struct.unpack('b', self.read(1))[0]

    def read_int16(self):
        if False:
            for i in range(10):
                print('nop')
        return struct.unpack('>h', self.read(2))[0]

    def read_int32(self):
        if False:
            return 10
        return struct.unpack('>i', self.read(4))[0]

    def read_int64(self):
        if False:
            return 10
        return struct.unpack('>q', self.read(8))[0]

    def read_float(self):
        if False:
            i = 10
            return i + 15
        return struct.unpack('>f', self.read(4))[0]

    def read_double(self):
        if False:
            return 10
        return struct.unpack('>d', self.read(8))[0]

    def read_bytes(self):
        if False:
            i = 10
            return i + 15
        size = self.read_int32()
        return self.read(size)

    def read_var_int64(self):
        if False:
            print('Hello World!')
        shift = 0
        result = 0
        while True:
            byte = self.read_byte()
            if byte < 0:
                raise RuntimeError('VarLong not terminated.')
            bits = byte & 127
            if shift >= 64 or (shift >= 63 and bits > 1):
                raise RuntimeError('VarLong too long.')
            result |= bits << shift
            shift += 7
            if not byte & 128:
                break
        if result >= 1 << 63:
            result -= 1 << 64
        return result

    def size(self):
        if False:
            i = 10
            return i + 15
        return len(self.data) - self.pos

class OutputStream(object):
    """
    A pure Python implementation of OutputStream.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = []
        self.byte_count = 0

    def write(self, b: bytes):
        if False:
            return 10
        self.data.append(b)
        self.byte_count += len(b)

    def write_byte(self, v):
        if False:
            return 10
        self.data.append(chr(v).encode('latin-1'))
        self.byte_count += 1

    def write_int8(self, v: int):
        if False:
            i = 10
            return i + 15
        self.write(struct.pack('b', v))

    def write_int16(self, v: int):
        if False:
            print('Hello World!')
        self.write(struct.pack('>h', v))

    def write_int32(self, v: int):
        if False:
            return 10
        self.write(struct.pack('>i', v))

    def write_int64(self, v: int):
        if False:
            return 10
        self.write(struct.pack('>q', v))

    def write_float(self, v: float):
        if False:
            for i in range(10):
                print('nop')
        self.write(struct.pack('>f', v))

    def write_double(self, v: float):
        if False:
            for i in range(10):
                print('nop')
        self.write(struct.pack('>d', v))

    def write_bytes(self, v: bytes, size: int):
        if False:
            print('Hello World!')
        self.write_int32(size)
        self.write(v[:size])

    def write_var_int64(self, v: int):
        if False:
            i = 10
            return i + 15
        if v < 0:
            v += 1 << 64
            if v <= 0:
                raise ValueError('Value too large (negative).')
        while True:
            bits = v & 127
            v >>= 7
            if v:
                bits |= 128
            self.data.append(chr(bits).encode('latin-1'))
            self.byte_count += 1
            if not v:
                break

    def get(self) -> bytes:
        if False:
            return 10
        return b''.join(self.data)

    def size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.byte_count

    def clear(self):
        if False:
            while True:
                i = 10
        self.data.clear()
        self.byte_count = 0