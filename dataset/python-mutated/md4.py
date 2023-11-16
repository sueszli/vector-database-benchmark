"""
MD4 implementation

Modified from:
https://gist.github.com/kangtastic/c3349fc4f9d659ee362b12d7d8c639b6
"""
import struct

class MD4:
    """
    An implementation of the MD4 hash algorithm.

    Modified to provide the same API as hashlib's.
    """
    name = 'md4'
    block_size = 64
    width = 32
    mask = 4294967295
    h = [1732584193, 4023233417, 2562383102, 271733878]

    def __init__(self, msg=b''):
        if False:
            for i in range(10):
                print('nop')
        self.msg = msg

    def update(self, msg):
        if False:
            while True:
                i = 10
        self.msg += msg

    def digest(self):
        if False:
            for i in range(10):
                print('nop')
        ml = len(self.msg) * 8
        self.msg += b'\x80'
        self.msg += b'\x00' * (-(len(self.msg) + 8) % self.block_size)
        self.msg += struct.pack('<Q', ml)
        self._process([self.msg[i:i + self.block_size] for i in range(0, len(self.msg), self.block_size)])
        return struct.pack('<4L', *self.h)

    def _process(self, chunks):
        if False:
            for i in range(10):
                print('nop')
        for chunk in chunks:
            (X, h) = (list(struct.unpack('<16I', chunk)), self.h.copy())
            Xi = [3, 7, 11, 19]
            for n in range(16):
                (i, j, k, l) = map(lambda x: x % 4, range(-n, -n + 4))
                (K, S) = (n, Xi[n % 4])
                hn = h[i] + MD4.F(h[j], h[k], h[l]) + X[K]
                h[i] = MD4.lrot(hn & MD4.mask, S)
            Xi = [3, 5, 9, 13]
            for n in range(16):
                (i, j, k, l) = map(lambda x: x % 4, range(-n, -n + 4))
                (K, S) = (n % 4 * 4 + n // 4, Xi[n % 4])
                hn = h[i] + MD4.G(h[j], h[k], h[l]) + X[K] + 1518500249
                h[i] = MD4.lrot(hn & MD4.mask, S)
            Xi = [3, 9, 11, 15]
            Ki = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
            for n in range(16):
                (i, j, k, l) = map(lambda x: x % 4, range(-n, -n + 4))
                (K, S) = (Ki[n], Xi[n % 4])
                hn = h[i] + MD4.H(h[j], h[k], h[l]) + X[K] + 1859775393
                h[i] = MD4.lrot(hn & MD4.mask, S)
            self.h = [v + n & MD4.mask for (v, n) in zip(self.h, h)]

    @staticmethod
    def F(x, y, z):
        if False:
            print('Hello World!')
        return x & y | ~x & z

    @staticmethod
    def G(x, y, z):
        if False:
            i = 10
            return i + 15
        return x & y | x & z | y & z

    @staticmethod
    def H(x, y, z):
        if False:
            return 10
        return x ^ y ^ z

    @staticmethod
    def lrot(value, n):
        if False:
            while True:
                i = 10
        (lbits, rbits) = (value << n & MD4.mask, value >> MD4.width - n)
        return lbits | rbits