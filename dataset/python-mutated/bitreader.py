class BitReaderError(Exception):
    pass

class _BitReader(object):

    def __init__(self, fileobj):
        if False:
            for i in range(10):
                print('nop')
        self._fileobj = fileobj
        self._buffer = 0
        self._bits = 0
        self._pos = fileobj.tell()

    def bits(self, count):
        if False:
            i = 10
            return i + 15
        'Reads `count` bits and returns an uint.\n\n        May raise BitReaderError if not enough data could be read or\n        IOError by the underlying file object.\n        '
        raise NotImplementedError

    def bytes(self, count):
        if False:
            i = 10
            return i + 15
        'Returns a bytearray of length `count`. Works unaligned.'
        if count < 0:
            raise ValueError
        if self._bits == 0:
            data = self._fileobj.read(count)
            if len(data) != count:
                raise BitReaderError('not enough data')
            return data
        return bytes(bytearray((self.bits(8) for _ in range(count))))

    def skip(self, count):
        if False:
            return 10
        "Skip `count` bits.\n\n        Might raise BitReaderError if there wasn't enough data to skip,\n        but might also fail on the next bits() instead.\n        "
        if count < 0:
            raise ValueError
        if count <= self._bits:
            self.bits(count)
        else:
            count -= self.align()
            n_bytes = count // 8
            self._fileobj.seek(n_bytes, 1)
            count -= n_bytes * 8
            self.bits(count)

    def get_position(self):
        if False:
            while True:
                i = 10
        'Returns the amount of bits read or skipped so far'
        return (self._fileobj.tell() - self._pos) * 8 - self._bits

    def align(self):
        if False:
            for i in range(10):
                print('nop')
        'Align to the next byte, returns the amount of bits skipped'
        bits = self._bits
        self._buffer = 0
        self._bits = 0
        return bits

    def is_aligned(self):
        if False:
            i = 10
            return i + 15
        'If we are currently aligned to bytes and nothing is buffered'
        return self._bits == 0

class MSBBitReader(_BitReader):
    """BitReader implementation which reads bits starting at LSB in each byte.
    """

    def bits(self, count):
        if False:
            for i in range(10):
                print('nop')
        'Reads `count` bits and returns an uint, MSB read first.\n\n        May raise BitReaderError if not enough data could be read or\n        IOError by the underlying file object.\n        '
        if count < 0:
            raise ValueError
        if count > self._bits:
            n_bytes = (count - self._bits + 7) // 8
            data = self._fileobj.read(n_bytes)
            if len(data) != n_bytes:
                raise BitReaderError('not enough data')
            for b in bytearray(data):
                self._buffer = self._buffer << 8 | b
            self._bits += n_bytes * 8
        self._bits -= count
        value = self._buffer >> self._bits
        self._buffer &= (1 << self._bits) - 1
        return value

class LSBBitReader(_BitReader):
    """BitReader implementation which reads bits starting at LSB in each byte.
    """

    def _lsb(self, count):
        if False:
            print('Hello World!')
        value = self._buffer & 255 >> 8 - count
        self._buffer = self._buffer >> count
        self._bits -= count
        return value

    def bits(self, count):
        if False:
            for i in range(10):
                print('nop')
        'Reads `count` bits and returns an uint, LSB read first.\n\n        May raise BitReaderError if not enough data could be read or\n        IOError by the underlying file object.\n        '
        if count < 0:
            raise ValueError
        value = 0
        if count <= self._bits:
            value = self._lsb(count)
        else:
            shift = 0
            remaining = count
            if self._bits > 0:
                remaining -= self._bits
                shift = self._bits
                value = self._lsb(self._bits)
            n_bytes = (remaining - self._bits + 7) // 8
            data = self._fileobj.read(n_bytes)
            if len(data) != n_bytes:
                raise BitReaderError('not enough data')
            for b in bytearray(data):
                if remaining > 8:
                    remaining -= 8
                    value = b << shift | value
                    shift += 8
                else:
                    self._buffer = b
                    self._bits = 8
                    b = self._lsb(remaining)
                    value = b << shift | value
        return value