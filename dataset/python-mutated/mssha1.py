"""
Modified version of SHA-1 used in Microsoft LIT files.

Adapted from the PyPy pure-Python SHA-1 implementation.
"""
__license__ = 'GPL v3'
__copyright__ = '2008, Marshall T. Vandegrift <llasram@gmail.com>'
import struct, copy
from polyglot.builtins import long_type

def _long2bytesBigEndian(n, blocksize=0):
    if False:
        while True:
            i = 10
    'Convert a long integer to a byte string.\n\n    If optional blocksize is given and greater than zero, pad the front\n    of the byte string with binary zeros so that the length is a multiple\n    of blocksize.\n    '
    s = b''
    pack = struct.pack
    while n > 0:
        s = pack('>I', n & 4294967295) + s
        n = n >> 32
    s = s.lstrip(b'\x00')
    if blocksize > 0 and len(s) % blocksize:
        s = (blocksize - len(s) % blocksize) * b'\x00' + s
    return s

def _bytelist2longBigEndian(blist):
    if False:
        print('Hello World!')
    'Transform a list of characters into a list of longs.'
    imax = len(blist) // 4
    hl = [0] * imax
    j = 0
    i = 0
    while i < imax:
        b0 = long_type(blist[j]) << 24
        b1 = long_type(blist[j + 1]) << 16
        b2 = long_type(blist[j + 2]) << 8
        b3 = long_type(blist[j + 3])
        hl[i] = b0 | b1 | b2 | b3
        i = i + 1
        j = j + 4
    return hl

def _rotateLeft(x, n):
    if False:
        i = 10
        return i + 15
    'Rotate x (32 bit) left n bits circular.'
    return x << n | x >> 32 - n

def f0_19(B, C, D):
    if False:
        for i in range(10):
            print('nop')
    return B & (C ^ D) ^ D

def f20_39(B, C, D):
    if False:
        return 10
    return B ^ C ^ D

def f40_59(B, C, D):
    if False:
        while True:
            i = 10
    return (B | C) & D | B & C

def f60_79(B, C, D):
    if False:
        print('Hello World!')
    return B ^ C ^ D

def f6_42(B, C, D):
    if False:
        for i in range(10):
            print('nop')
    return B + C ^ C
f = [f0_19] * 20 + [f20_39] * 20 + [f40_59] * 20 + [f60_79] * 20
f[3] = f20_39
f[6] = f6_42
f[10] = f20_39
f[15] = f20_39
f[26] = f0_19
f[31] = f40_59
f[42] = f6_42
f[51] = f20_39
f[68] = f0_19
K = [1518500249, 1859775393, 2400959708, 3395469782]

class mssha1:
    """An implementation of the MD5 hash function in pure Python."""

    def __init__(self):
        if False:
            return 10
        'Initialisation.'
        self.length = 0
        self.count = [0, 0]
        self.input = bytearray()
        self.init()

    def init(self):
        if False:
            print('Hello World!')
        'Initialize the message-digest and set all fields to zero.'
        self.length = 0
        self.input = []
        self.H0 = 839939668
        self.H1 = 587294533
        self.H2 = 3303440546
        self.H3 = 3697776675
        self.H4 = 3498408500

    def _transform(self, W):
        if False:
            i = 10
            return i + 15
        for t in range(16, 80):
            W.append(_rotateLeft(W[t - 3] ^ W[t - 8] ^ W[t - 14] ^ W[t - 16], 1) & 4294967295)
        A = self.H0
        B = self.H1
        C = self.H2
        D = self.H3
        E = self.H4
        for t in range(0, 80):
            TEMP = _rotateLeft(A, 5) + f[t](B, C, D) + E + W[t] + K[t // 20]
            E = D
            D = C
            C = _rotateLeft(B, 30) & 4294967295
            B = A
            A = TEMP & 4294967295
        self.H0 = self.H0 + A & 4294967295
        self.H1 = self.H1 + B & 4294967295
        self.H2 = self.H2 + C & 4294967295
        self.H3 = self.H3 + D & 4294967295
        self.H4 = self.H4 + E & 4294967295

    def update(self, inBuf):
        if False:
            print('Hello World!')
        'Add to the current message.\n\n        Update the mssha1 object with the string arg. Repeated calls\n        are equivalent to a single call with the concatenation of all\n        the arguments, i.e. s.update(a); s.update(b) is equivalent\n        to s.update(a+b).\n\n        The hash is immediately calculated for all full blocks. The final\n        calculation is made in digest(). It will calculate 1-2 blocks,\n        depending on how much padding we have to add. This allows us to\n        keep an intermediate value for the hash, so that we only need to\n        make minimal recalculation if we call update() to add more data\n        to the hashed string.\n        '
        inBuf = bytearray(inBuf)
        leninBuf = long_type(len(inBuf))
        index = self.count[1] >> 3 & 63
        self.count[1] = self.count[1] + (leninBuf << 3)
        if self.count[1] < leninBuf << 3:
            self.count[0] = self.count[0] + 1
        self.count[0] = self.count[0] + (leninBuf >> 29)
        partLen = 64 - index
        if leninBuf >= partLen:
            self.input[index:] = inBuf[:partLen]
            self._transform(_bytelist2longBigEndian(self.input))
            i = partLen
            while i + 63 < leninBuf:
                self._transform(_bytelist2longBigEndian(inBuf[i:i + 64]))
                i = i + 64
            else:
                self.input = inBuf[i:leninBuf]
        else:
            i = 0
            self.input = self.input + inBuf

    def digest(self):
        if False:
            print('Hello World!')
        'Terminate the message-digest computation and return digest.\n\n        Return the digest of the strings passed to the update()\n        method so far. This is a 16-byte string which may contain\n        non-ASCII characters, including null bytes.\n        '
        H0 = self.H0
        H1 = self.H1
        H2 = self.H2
        H3 = self.H3
        H4 = self.H4
        inp = bytearray(self.input)
        count = [] + self.count
        index = self.count[1] >> 3 & 63
        if index < 56:
            padLen = 56 - index
        else:
            padLen = 120 - index
        padding = b'\x80' + b'\x00' * 63
        self.update(padding[:padLen])
        bits = _bytelist2longBigEndian(self.input[:56]) + count
        self._transform(bits)
        digest = _long2bytesBigEndian(self.H0, 4) + _long2bytesBigEndian(self.H1, 4) + _long2bytesBigEndian(self.H2, 4) + _long2bytesBigEndian(self.H3, 4) + _long2bytesBigEndian(self.H4, 4)
        self.H0 = H0
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.H4 = H4
        self.input = inp
        self.count = count
        return digest

    def hexdigest(self):
        if False:
            for i in range(10):
                print('nop')
        'Terminate and return digest in HEX form.\n\n        Like digest() except the digest is returned as a string of\n        length 32, containing only hexadecimal digits. This may be\n        used to exchange the value safely in email or other non-\n        binary environments.\n        '
        return ''.join(['%02x' % c for c in bytearray(self.digest())])

    def copy(self):
        if False:
            print('Hello World!')
        "Return a clone object.\n\n        Return a copy ('clone') of the md5 object. This can be used\n        to efficiently compute the digests of strings that share\n        a common initial substring.\n        "
        return copy.deepcopy(self)
digest_size = digestsize = 20
blocksize = 1

def new(arg=None):
    if False:
        return 10
    'Return a new mssha1 crypto object.\n\n    If arg is present, the method call update(arg) is made.\n    '
    crypto = mssha1()
    if arg:
        crypto.update(arg)
    return crypto
if __name__ == '__main__':

    def main():
        if False:
            i = 10
            return i + 15
        import sys
        file = None
        if len(sys.argv) > 2:
            print('usage: %s [FILE]' % sys.argv[0])
            return
        elif len(sys.argv) < 2:
            file = sys.stdin
        else:
            file = open(sys.argv[1], 'rb')
        context = new()
        data = file.read(16384)
        while data:
            context.update(data)
            data = file.read(16384)
        file.close()
        digest = context.hexdigest().upper()
        for i in range(0, 40, 8):
            print(digest[i:i + 8], end=' ')
        print()
    main()