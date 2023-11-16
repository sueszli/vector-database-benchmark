import struct

class UInt64:

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value & 18446744073709551615

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        return self._value

    @value.setter
    def value(self, new_value):
        if False:
            i = 10
            return i + 15
        self._value = new_value & 18446744073709551615

    def __add__(self, other):
        if False:
            return 10
        result = self.value + (other.value if isinstance(other, UInt64) else other)
        return UInt64(result & 18446744073709551615)

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        result = self.value - (other.value if isinstance(other, UInt64) else other)
        return UInt64(result & 18446744073709551615)

    def __mul__(self, other):
        if False:
            return 10
        result = self.value * (other.value if isinstance(other, UInt64) else other)
        return UInt64(result & 18446744073709551615)

    def __xor__(self, other):
        if False:
            i = 10
            return i + 15
        result = self.value ^ (other.value if isinstance(other, UInt64) else other)
        return UInt64(result & 18446744073709551615)

    def __lshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        result = self.value << (other.value if isinstance(other, UInt64) else other)
        return UInt64(result & 18446744073709551615)

    def __rshift__(self, other):
        if False:
            i = 10
            return i + 15
        result = self.value >> (other.value if isinstance(other, UInt64) else other)
        return UInt64(result & 18446744073709551615)

    def __and__(self, other):
        if False:
            print('Hello World!')
        result = self.value & (other.value if isinstance(other, UInt64) else other)
        return UInt64(result & 18446744073709551615)

    def __or__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, UInt64):
            return UInt64(self.value | other.value)
        elif isinstance(other, int):
            return UInt64(self.value | other)
        else:
            raise TypeError('Unsupported type for OR operation')

    def __invert__(self):
        if False:
            while True:
                i = 10
        return UInt64(~self.value & 18446744073709551615)

    @staticmethod
    def encode_double_as_uint64(value):
        if False:
            return 10
        value_in_uint64 = struct.unpack('<Q', struct.pack('<d', value))[0]
        mask = 9223372036854775808
        return value_in_uint64 ^ mask if value_in_uint64 < mask else ~value_in_uint64 + 1

    @staticmethod
    def decode_double_from_uint64(value):
        if False:
            while True:
                i = 10
        mask = 9223372036854775808
        value = ~(value - 1) if value < mask else value ^ mask
        return struct.unpack('<d', struct.pack('<Q', value))[0]

    def __int__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

class UInt128:

    def __init__(self, low, high):
        if False:
            print('Hello World!')
        if isinstance(low, UInt64):
            self.low = low
        else:
            self.low = UInt64(low)
        if isinstance(high, UInt64):
            self.high = high
        else:
            self.high = UInt64(high)

    def __add__(self, other):
        if False:
            print('Hello World!')
        low = self.low + other.low
        high = self.high + other.high + UInt64(int(low.value > 18446744073709551615))
        return UInt128(low & 18446744073709551615, high & 18446744073709551615)

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        borrow = UInt64(0)
        if self.low < other.low:
            borrow = UInt64(1)
        low = self.low - other.low & 18446744073709551615
        high = self.high - other.high - borrow & 18446744073709551615
        return UInt128(low, high)

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        pass

    def __xor__(self, other):
        if False:
            return 10
        low = self.low ^ other.low
        high = self.high ^ other.high
        return UInt128(low, high)

    def __and__(self, other):
        if False:
            for i in range(10):
                print('nop')
        low = self.low & other.low
        high = self.high & other.high
        return UInt128(low, high)

    def __or__(self, other):
        if False:
            while True:
                i = 10
        low = self.low | other.low
        high = self.high | other.high
        return UInt128(low, high)

    def __lshift__(self, shift):
        if False:
            return 10
        pass

    def __rshift__(self, shift):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_low(self):
        if False:
            return 10
        return self.low

    def get_high(self):
        if False:
            print('Hello World!')
        return self.high

    def as_tuple(self):
        if False:
            return 10
        return (self.low.value, self.high.value)

    def as_hex(self):
        if False:
            while True:
                i = 10
        return hex(self.high.value)[2:].zfill(16) + hex(self.low.value)[2:].zfill(16)

    def as_int(self):
        if False:
            print('Hello World!')
        return self.high.value << 64 | self.low.value

    def __str__(self):
        if False:
            return 10
        return str(self.as_int())

    def to_byte_array(self):
        if False:
            i = 10
            return i + 15
        high_bytes = self.high.value.to_bytes(8, byteorder='little')
        low_bytes = self.low.value.to_bytes(8, byteorder='little')
        byte_array = bytearray(low_bytes + high_bytes)
        return byte_array

    @staticmethod
    def create(low, high):
        if False:
            i = 10
            return i + 15
        return UInt128(low, high)