from builtins import str
from future.utils import PY3
from miasm.core.utils import BIG_ENDIAN, LITTLE_ENDIAN
from miasm.core.utils import upck8le, upck16le, upck32le, upck64le
from miasm.core.utils import upck8be, upck16be, upck32be, upck64be

class bin_stream(object):
    _cache = None
    CACHE_SIZE = 10000
    _atomic_mode = False

    def __init__(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        self.endianness = LITTLE_ENDIAN

    def __repr__(self):
        if False:
            return 10
        return '<%s !!>' % self.__class__.__name__

    def __str__(self):
        if False:
            return 10
        if PY3:
            return repr(self)
        return self.__bytes__()

    def hexdump(self, offset, l):
        if False:
            while True:
                i = 10
        return

    def enter_atomic_mode(self):
        if False:
            i = 10
            return i + 15
        'Enter atomic mode. In this mode, read may be cached'
        assert not self._atomic_mode
        self._atomic_mode = True
        self._cache = {}

    def leave_atomic_mode(self):
        if False:
            print('Hello World!')
        'Leave atomic mode'
        assert self._atomic_mode
        self._atomic_mode = False
        self._cache = None

    def _getbytes(self, start, length):
        if False:
            i = 10
            return i + 15
        return self.bin[start:start + length]

    def getbytes(self, start, l=1):
        if False:
            i = 10
            return i + 15
        'Return the bytes from the bit stream\n        @start: starting offset (in byte)\n        @l: (optional) number of bytes to read\n\n        Wrapper on _getbytes, with atomic mode handling.\n        '
        if self._atomic_mode:
            val = self._cache.get((start, l), None)
            if val is None:
                val = self._getbytes(start, l)
                self._cache[start, l] = val
        else:
            val = self._getbytes(start, l)
        return val

    def getbits(self, start, n):
        if False:
            print('Hello World!')
        'Return the bits from the bit stream\n        @start: the offset in bits\n        @n: number of bits to read\n        '
        if n == 0:
            return 0
        if n > self.getlen() * 8:
            raise IOError('not enough bits %r %r' % (n, len(self.bin) * 8))
        byte_start = start // 8
        byte_stop = (start + n + 7) // 8
        temp = self.getbytes(byte_start, byte_stop - byte_start)
        if not temp:
            raise IOError('cannot get bytes')
        start = start % 8
        out = 0
        while n:
            cur_byte_idx = start // 8
            new_bits = ord(temp[cur_byte_idx:cur_byte_idx + 1])
            to_keep = 8 - start % 8
            new_bits &= (1 << to_keep) - 1
            cur_len = min(to_keep, n)
            new_bits >>= to_keep - cur_len
            out <<= cur_len
            out |= new_bits
            n -= cur_len
            start += cur_len
        return out

    def get_u8(self, addr, endianness=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return u8 from address @addr\n        endianness: Optional: LITTLE_ENDIAN/BIG_ENDIAN\n        '
        if endianness is None:
            endianness = self.endianness
        data = self.getbytes(addr, 1)
        return data

    def get_u16(self, addr, endianness=None):
        if False:
            while True:
                i = 10
        '\n        Return u16 from address @addr\n        endianness: Optional: LITTLE_ENDIAN/BIG_ENDIAN\n        '
        if endianness is None:
            endianness = self.endianness
        data = self.getbytes(addr, 2)
        if endianness == LITTLE_ENDIAN:
            return upck16le(data)
        else:
            return upck16be(data)

    def get_u32(self, addr, endianness=None):
        if False:
            print('Hello World!')
        '\n        Return u32 from address @addr\n        endianness: Optional: LITTLE_ENDIAN/BIG_ENDIAN\n        '
        if endianness is None:
            endianness = self.endianness
        data = self.getbytes(addr, 4)
        if endianness == LITTLE_ENDIAN:
            return upck32le(data)
        else:
            return upck32be(data)

    def get_u64(self, addr, endianness=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return u64 from address @addr\n        endianness: Optional: LITTLE_ENDIAN/BIG_ENDIAN\n        '
        if endianness is None:
            endianness = self.endianness
        data = self.getbytes(addr, 8)
        if endianness == LITTLE_ENDIAN:
            return upck64le(data)
        else:
            return upck64be(data)

class bin_stream_str(bin_stream):

    def __init__(self, input_str=b'', offset=0, base_address=0, shift=None):
        if False:
            while True:
                i = 10
        bin_stream.__init__(self)
        if shift is not None:
            raise DeprecationWarning('use base_address instead of shift')
        self.bin = input_str
        self.offset = offset
        self.base_address = base_address
        self.l = len(input_str)

    def _getbytes(self, start, l=1):
        if False:
            for i in range(10):
                print('nop')
        if start + l - self.base_address > self.l:
            raise IOError('not enough bytes in str')
        if start - self.base_address < 0:
            raise IOError('Negative offset')
        return super(bin_stream_str, self)._getbytes(start - self.base_address, l)

    def readbs(self, l=1):
        if False:
            print('Hello World!')
        if self.offset + l - self.base_address > self.l:
            raise IOError('not enough bytes in str')
        if self.offset - self.base_address < 0:
            raise IOError('Negative offset')
        self.offset += l
        return self.bin[self.offset - l - self.base_address:self.offset - self.base_address]

    def __bytes__(self):
        if False:
            while True:
                i = 10
        return self.bin[self.offset - self.base_address:]

    def setoffset(self, val):
        if False:
            return 10
        self.offset = val

    def getlen(self):
        if False:
            i = 10
            return i + 15
        return self.l - (self.offset - self.base_address)

class bin_stream_file(bin_stream):

    def __init__(self, binary, offset=0, base_address=0, shift=None):
        if False:
            i = 10
            return i + 15
        bin_stream.__init__(self)
        if shift is not None:
            raise DeprecationWarning('use base_address instead of shift')
        self.bin = binary
        self.bin.seek(0, 2)
        self.base_address = base_address
        self.l = self.bin.tell()
        self.offset = offset

    def getoffset(self):
        if False:
            i = 10
            return i + 15
        return self.bin.tell() + self.base_address

    def setoffset(self, val):
        if False:
            print('Hello World!')
        self.bin.seek(val - self.base_address)
    offset = property(getoffset, setoffset)

    def readbs(self, l=1):
        if False:
            while True:
                i = 10
        if self.offset + l - self.base_address > self.l:
            raise IOError('not enough bytes in file')
        if self.offset - self.base_address < 0:
            raise IOError('Negative offset')
        return self.bin.read(l)

    def __bytes__(self):
        if False:
            while True:
                i = 10
        return self.bin.read()

    def getlen(self):
        if False:
            return 10
        return self.l - (self.offset - self.base_address)

class bin_stream_container(bin_stream):

    def __init__(self, binary, offset=0):
        if False:
            for i in range(10):
                print('nop')
        bin_stream.__init__(self)
        self.bin = binary
        self.l = binary.virt.max_addr()
        self.offset = offset

    def is_addr_in(self, ad):
        if False:
            for i in range(10):
                print('nop')
        return self.bin.virt.is_addr_in(ad)

    def getlen(self):
        if False:
            while True:
                i = 10
        return self.l

    def readbs(self, l=1):
        if False:
            i = 10
            return i + 15
        if self.offset + l > self.l:
            raise IOError('not enough bytes')
        if self.offset < 0:
            raise IOError('Negative offset')
        self.offset += l
        return self.bin.virt.get(self.offset - l, self.offset)

    def _getbytes(self, start, l=1):
        if False:
            return 10
        try:
            return self.bin.virt.get(start, start + l)
        except ValueError:
            raise IOError('cannot get bytes')

    def __bytes__(self):
        if False:
            while True:
                i = 10
        return self.bin.virt.get(self.offset, self.offset + self.l)

    def setoffset(self, val):
        if False:
            return 10
        self.offset = val

class bin_stream_pe(bin_stream_container):

    def __init__(self, binary, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(bin_stream_pe, self).__init__(binary, *args, **kwargs)
        self.endianness = binary._sex

class bin_stream_elf(bin_stream_container):

    def __init__(self, binary, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(bin_stream_elf, self).__init__(binary, *args, **kwargs)
        self.endianness = binary.sex

class bin_stream_vm(bin_stream):

    def __init__(self, vm, offset=0, base_offset=0):
        if False:
            print('Hello World!')
        self.offset = offset
        self.base_offset = base_offset
        self.vm = vm
        if self.vm.is_little_endian():
            self.endianness = LITTLE_ENDIAN
        else:
            self.endianness = BIG_ENDIAN

    def getlen(self):
        if False:
            while True:
                i = 10
        return 18446744073709551615

    def _getbytes(self, start, l=1):
        if False:
            print('Hello World!')
        try:
            s = self.vm.get_mem(start + self.base_offset, l)
        except:
            raise IOError('cannot get mem ad', hex(start))
        return s

    def readbs(self, l=1):
        if False:
            while True:
                i = 10
        try:
            s = self.vm.get_mem(self.offset + self.base_offset, l)
        except:
            raise IOError('cannot get mem ad', hex(self.offset))
        self.offset += l
        return s

    def setoffset(self, val):
        if False:
            for i in range(10):
                print('nop')
        self.offset = val