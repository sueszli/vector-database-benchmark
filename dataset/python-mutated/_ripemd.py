import sys
digest_size = 20
digestsize = 20

class RIPEMD160:
    """
    Return a new RIPEMD160 object. An optional string argument
    may be provided; if present, this string will be automatically
    hashed.
    """

    def __init__(self, arg=None):
        if False:
            i = 10
            return i + 15
        self.ctx = RMDContext()
        if arg:
            self.update(arg)
        self.dig = None

    def update(self, arg):
        if False:
            i = 10
            return i + 15
        RMD160Update(self.ctx, arg, len(arg))
        self.dig = None

    def digest(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dig:
            return self.dig
        ctx = self.ctx.copy()
        self.dig = RMD160Final(self.ctx)
        self.ctx = ctx
        return self.dig

    def hexdigest(self):
        if False:
            for i in range(10):
                print('nop')
        dig = self.digest()
        hex_digest = ''
        for d in dig:
            hex_digest += '%02x' % d
        return hex_digest

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        import copy
        return copy.deepcopy(self)

def new(arg=None):
    if False:
        return 10
    '\n    Return a new RIPEMD160 object. An optional string argument\n    may be provided; if present, this string will be automatically\n    hashed.\n    '
    return RIPEMD160(arg)

class RMDContext:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.state = [1732584193, 4023233417, 2562383102, 271733878, 3285377520]
        self.count = 0
        self.buffer = [0] * 64

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        ctx = RMDContext()
        ctx.state = self.state[:]
        ctx.count = self.count
        ctx.buffer = self.buffer[:]
        return ctx
K0 = 0
K1 = 1518500249
K2 = 1859775393
K3 = 2400959708
K4 = 2840853838
KK0 = 1352829926
KK1 = 1548603684
KK2 = 1836072691
KK3 = 2053994217
KK4 = 0

def ROL(n, x):
    if False:
        return 10
    return x << n & 4294967295 | x >> 32 - n

def F0(x, y, z):
    if False:
        return 10
    return x ^ y ^ z

def F1(x, y, z):
    if False:
        i = 10
        return i + 15
    return x & y | ~x % 4294967296 & z

def F2(x, y, z):
    if False:
        return 10
    return (x | ~y % 4294967296) ^ z

def F3(x, y, z):
    if False:
        while True:
            i = 10
    return x & z | ~z % 4294967296 & y

def F4(x, y, z):
    if False:
        return 10
    return x ^ (y | ~z % 4294967296)

def R(a, b, c, d, e, Fj, Kj, sj, rj, X):
    if False:
        print('Hello World!')
    a = ROL(sj, (a + Fj(b, c, d) + X[rj] + Kj) % 4294967296) + e
    c = ROL(10, c)
    return (a % 4294967296, c)
PADDING = [128] + [0] * 63
import sys
import struct

def RMD160Transform(state, block):
    if False:
        print('Hello World!')
    x = [0] * 16
    if sys.byteorder == 'little':
        x = struct.unpack('<16L', bytes(block[0:64]))
    else:
        raise ValueError('Big-endian platforms are not supported')
    a = state[0]
    b = state[1]
    c = state[2]
    d = state[3]
    e = state[4]
    (a, c) = R(a, b, c, d, e, F0, K0, 11, 0, x)
    (e, b) = R(e, a, b, c, d, F0, K0, 14, 1, x)
    (d, a) = R(d, e, a, b, c, F0, K0, 15, 2, x)
    (c, e) = R(c, d, e, a, b, F0, K0, 12, 3, x)
    (b, d) = R(b, c, d, e, a, F0, K0, 5, 4, x)
    (a, c) = R(a, b, c, d, e, F0, K0, 8, 5, x)
    (e, b) = R(e, a, b, c, d, F0, K0, 7, 6, x)
    (d, a) = R(d, e, a, b, c, F0, K0, 9, 7, x)
    (c, e) = R(c, d, e, a, b, F0, K0, 11, 8, x)
    (b, d) = R(b, c, d, e, a, F0, K0, 13, 9, x)
    (a, c) = R(a, b, c, d, e, F0, K0, 14, 10, x)
    (e, b) = R(e, a, b, c, d, F0, K0, 15, 11, x)
    (d, a) = R(d, e, a, b, c, F0, K0, 6, 12, x)
    (c, e) = R(c, d, e, a, b, F0, K0, 7, 13, x)
    (b, d) = R(b, c, d, e, a, F0, K0, 9, 14, x)
    (a, c) = R(a, b, c, d, e, F0, K0, 8, 15, x)
    (e, b) = R(e, a, b, c, d, F1, K1, 7, 7, x)
    (d, a) = R(d, e, a, b, c, F1, K1, 6, 4, x)
    (c, e) = R(c, d, e, a, b, F1, K1, 8, 13, x)
    (b, d) = R(b, c, d, e, a, F1, K1, 13, 1, x)
    (a, c) = R(a, b, c, d, e, F1, K1, 11, 10, x)
    (e, b) = R(e, a, b, c, d, F1, K1, 9, 6, x)
    (d, a) = R(d, e, a, b, c, F1, K1, 7, 15, x)
    (c, e) = R(c, d, e, a, b, F1, K1, 15, 3, x)
    (b, d) = R(b, c, d, e, a, F1, K1, 7, 12, x)
    (a, c) = R(a, b, c, d, e, F1, K1, 12, 0, x)
    (e, b) = R(e, a, b, c, d, F1, K1, 15, 9, x)
    (d, a) = R(d, e, a, b, c, F1, K1, 9, 5, x)
    (c, e) = R(c, d, e, a, b, F1, K1, 11, 2, x)
    (b, d) = R(b, c, d, e, a, F1, K1, 7, 14, x)
    (a, c) = R(a, b, c, d, e, F1, K1, 13, 11, x)
    (e, b) = R(e, a, b, c, d, F1, K1, 12, 8, x)
    (d, a) = R(d, e, a, b, c, F2, K2, 11, 3, x)
    (c, e) = R(c, d, e, a, b, F2, K2, 13, 10, x)
    (b, d) = R(b, c, d, e, a, F2, K2, 6, 14, x)
    (a, c) = R(a, b, c, d, e, F2, K2, 7, 4, x)
    (e, b) = R(e, a, b, c, d, F2, K2, 14, 9, x)
    (d, a) = R(d, e, a, b, c, F2, K2, 9, 15, x)
    (c, e) = R(c, d, e, a, b, F2, K2, 13, 8, x)
    (b, d) = R(b, c, d, e, a, F2, K2, 15, 1, x)
    (a, c) = R(a, b, c, d, e, F2, K2, 14, 2, x)
    (e, b) = R(e, a, b, c, d, F2, K2, 8, 7, x)
    (d, a) = R(d, e, a, b, c, F2, K2, 13, 0, x)
    (c, e) = R(c, d, e, a, b, F2, K2, 6, 6, x)
    (b, d) = R(b, c, d, e, a, F2, K2, 5, 13, x)
    (a, c) = R(a, b, c, d, e, F2, K2, 12, 11, x)
    (e, b) = R(e, a, b, c, d, F2, K2, 7, 5, x)
    (d, a) = R(d, e, a, b, c, F2, K2, 5, 12, x)
    (c, e) = R(c, d, e, a, b, F3, K3, 11, 1, x)
    (b, d) = R(b, c, d, e, a, F3, K3, 12, 9, x)
    (a, c) = R(a, b, c, d, e, F3, K3, 14, 11, x)
    (e, b) = R(e, a, b, c, d, F3, K3, 15, 10, x)
    (d, a) = R(d, e, a, b, c, F3, K3, 14, 0, x)
    (c, e) = R(c, d, e, a, b, F3, K3, 15, 8, x)
    (b, d) = R(b, c, d, e, a, F3, K3, 9, 12, x)
    (a, c) = R(a, b, c, d, e, F3, K3, 8, 4, x)
    (e, b) = R(e, a, b, c, d, F3, K3, 9, 13, x)
    (d, a) = R(d, e, a, b, c, F3, K3, 14, 3, x)
    (c, e) = R(c, d, e, a, b, F3, K3, 5, 7, x)
    (b, d) = R(b, c, d, e, a, F3, K3, 6, 15, x)
    (a, c) = R(a, b, c, d, e, F3, K3, 8, 14, x)
    (e, b) = R(e, a, b, c, d, F3, K3, 6, 5, x)
    (d, a) = R(d, e, a, b, c, F3, K3, 5, 6, x)
    (c, e) = R(c, d, e, a, b, F3, K3, 12, 2, x)
    (b, d) = R(b, c, d, e, a, F4, K4, 9, 4, x)
    (a, c) = R(a, b, c, d, e, F4, K4, 15, 0, x)
    (e, b) = R(e, a, b, c, d, F4, K4, 5, 5, x)
    (d, a) = R(d, e, a, b, c, F4, K4, 11, 9, x)
    (c, e) = R(c, d, e, a, b, F4, K4, 6, 7, x)
    (b, d) = R(b, c, d, e, a, F4, K4, 8, 12, x)
    (a, c) = R(a, b, c, d, e, F4, K4, 13, 2, x)
    (e, b) = R(e, a, b, c, d, F4, K4, 12, 10, x)
    (d, a) = R(d, e, a, b, c, F4, K4, 5, 14, x)
    (c, e) = R(c, d, e, a, b, F4, K4, 12, 1, x)
    (b, d) = R(b, c, d, e, a, F4, K4, 13, 3, x)
    (a, c) = R(a, b, c, d, e, F4, K4, 14, 8, x)
    (e, b) = R(e, a, b, c, d, F4, K4, 11, 11, x)
    (d, a) = R(d, e, a, b, c, F4, K4, 8, 6, x)
    (c, e) = R(c, d, e, a, b, F4, K4, 5, 15, x)
    (b, d) = R(b, c, d, e, a, F4, K4, 6, 13, x)
    aa = a
    bb = b
    cc = c
    dd = d
    ee = e
    a = state[0]
    b = state[1]
    c = state[2]
    d = state[3]
    e = state[4]
    (a, c) = R(a, b, c, d, e, F4, KK0, 8, 5, x)
    (e, b) = R(e, a, b, c, d, F4, KK0, 9, 14, x)
    (d, a) = R(d, e, a, b, c, F4, KK0, 9, 7, x)
    (c, e) = R(c, d, e, a, b, F4, KK0, 11, 0, x)
    (b, d) = R(b, c, d, e, a, F4, KK0, 13, 9, x)
    (a, c) = R(a, b, c, d, e, F4, KK0, 15, 2, x)
    (e, b) = R(e, a, b, c, d, F4, KK0, 15, 11, x)
    (d, a) = R(d, e, a, b, c, F4, KK0, 5, 4, x)
    (c, e) = R(c, d, e, a, b, F4, KK0, 7, 13, x)
    (b, d) = R(b, c, d, e, a, F4, KK0, 7, 6, x)
    (a, c) = R(a, b, c, d, e, F4, KK0, 8, 15, x)
    (e, b) = R(e, a, b, c, d, F4, KK0, 11, 8, x)
    (d, a) = R(d, e, a, b, c, F4, KK0, 14, 1, x)
    (c, e) = R(c, d, e, a, b, F4, KK0, 14, 10, x)
    (b, d) = R(b, c, d, e, a, F4, KK0, 12, 3, x)
    (a, c) = R(a, b, c, d, e, F4, KK0, 6, 12, x)
    (e, b) = R(e, a, b, c, d, F3, KK1, 9, 6, x)
    (d, a) = R(d, e, a, b, c, F3, KK1, 13, 11, x)
    (c, e) = R(c, d, e, a, b, F3, KK1, 15, 3, x)
    (b, d) = R(b, c, d, e, a, F3, KK1, 7, 7, x)
    (a, c) = R(a, b, c, d, e, F3, KK1, 12, 0, x)
    (e, b) = R(e, a, b, c, d, F3, KK1, 8, 13, x)
    (d, a) = R(d, e, a, b, c, F3, KK1, 9, 5, x)
    (c, e) = R(c, d, e, a, b, F3, KK1, 11, 10, x)
    (b, d) = R(b, c, d, e, a, F3, KK1, 7, 14, x)
    (a, c) = R(a, b, c, d, e, F3, KK1, 7, 15, x)
    (e, b) = R(e, a, b, c, d, F3, KK1, 12, 8, x)
    (d, a) = R(d, e, a, b, c, F3, KK1, 7, 12, x)
    (c, e) = R(c, d, e, a, b, F3, KK1, 6, 4, x)
    (b, d) = R(b, c, d, e, a, F3, KK1, 15, 9, x)
    (a, c) = R(a, b, c, d, e, F3, KK1, 13, 1, x)
    (e, b) = R(e, a, b, c, d, F3, KK1, 11, 2, x)
    (d, a) = R(d, e, a, b, c, F2, KK2, 9, 15, x)
    (c, e) = R(c, d, e, a, b, F2, KK2, 7, 5, x)
    (b, d) = R(b, c, d, e, a, F2, KK2, 15, 1, x)
    (a, c) = R(a, b, c, d, e, F2, KK2, 11, 3, x)
    (e, b) = R(e, a, b, c, d, F2, KK2, 8, 7, x)
    (d, a) = R(d, e, a, b, c, F2, KK2, 6, 14, x)
    (c, e) = R(c, d, e, a, b, F2, KK2, 6, 6, x)
    (b, d) = R(b, c, d, e, a, F2, KK2, 14, 9, x)
    (a, c) = R(a, b, c, d, e, F2, KK2, 12, 11, x)
    (e, b) = R(e, a, b, c, d, F2, KK2, 13, 8, x)
    (d, a) = R(d, e, a, b, c, F2, KK2, 5, 12, x)
    (c, e) = R(c, d, e, a, b, F2, KK2, 14, 2, x)
    (b, d) = R(b, c, d, e, a, F2, KK2, 13, 10, x)
    (a, c) = R(a, b, c, d, e, F2, KK2, 13, 0, x)
    (e, b) = R(e, a, b, c, d, F2, KK2, 7, 4, x)
    (d, a) = R(d, e, a, b, c, F2, KK2, 5, 13, x)
    (c, e) = R(c, d, e, a, b, F1, KK3, 15, 8, x)
    (b, d) = R(b, c, d, e, a, F1, KK3, 5, 6, x)
    (a, c) = R(a, b, c, d, e, F1, KK3, 8, 4, x)
    (e, b) = R(e, a, b, c, d, F1, KK3, 11, 1, x)
    (d, a) = R(d, e, a, b, c, F1, KK3, 14, 3, x)
    (c, e) = R(c, d, e, a, b, F1, KK3, 14, 11, x)
    (b, d) = R(b, c, d, e, a, F1, KK3, 6, 15, x)
    (a, c) = R(a, b, c, d, e, F1, KK3, 14, 0, x)
    (e, b) = R(e, a, b, c, d, F1, KK3, 6, 5, x)
    (d, a) = R(d, e, a, b, c, F1, KK3, 9, 12, x)
    (c, e) = R(c, d, e, a, b, F1, KK3, 12, 2, x)
    (b, d) = R(b, c, d, e, a, F1, KK3, 9, 13, x)
    (a, c) = R(a, b, c, d, e, F1, KK3, 12, 9, x)
    (e, b) = R(e, a, b, c, d, F1, KK3, 5, 7, x)
    (d, a) = R(d, e, a, b, c, F1, KK3, 15, 10, x)
    (c, e) = R(c, d, e, a, b, F1, KK3, 8, 14, x)
    (b, d) = R(b, c, d, e, a, F0, KK4, 8, 12, x)
    (a, c) = R(a, b, c, d, e, F0, KK4, 5, 15, x)
    (e, b) = R(e, a, b, c, d, F0, KK4, 12, 10, x)
    (d, a) = R(d, e, a, b, c, F0, KK4, 9, 4, x)
    (c, e) = R(c, d, e, a, b, F0, KK4, 12, 1, x)
    (b, d) = R(b, c, d, e, a, F0, KK4, 5, 5, x)
    (a, c) = R(a, b, c, d, e, F0, KK4, 14, 8, x)
    (e, b) = R(e, a, b, c, d, F0, KK4, 6, 7, x)
    (d, a) = R(d, e, a, b, c, F0, KK4, 8, 6, x)
    (c, e) = R(c, d, e, a, b, F0, KK4, 13, 2, x)
    (b, d) = R(b, c, d, e, a, F0, KK4, 6, 13, x)
    (a, c) = R(a, b, c, d, e, F0, KK4, 5, 14, x)
    (e, b) = R(e, a, b, c, d, F0, KK4, 15, 0, x)
    (d, a) = R(d, e, a, b, c, F0, KK4, 13, 3, x)
    (c, e) = R(c, d, e, a, b, F0, KK4, 11, 9, x)
    (b, d) = R(b, c, d, e, a, F0, KK4, 11, 11, x)
    t = (state[1] + cc + d) % 4294967296
    state[1] = (state[2] + dd + e) % 4294967296
    state[2] = (state[3] + ee + a) % 4294967296
    state[3] = (state[4] + aa + b) % 4294967296
    state[4] = (state[0] + bb + c) % 4294967296
    state[0] = t % 4294967296

def RMD160Update(ctx, inp, inplen):
    if False:
        print('Hello World!')
    if type(inp) == str:
        inp = [ord(i) & 255 for i in inp]
    have = int(ctx.count // 8 % 64)
    inplen = int(inplen)
    need = 64 - have
    ctx.count += 8 * inplen
    off = 0
    if inplen >= need:
        if have:
            for i in range(need):
                ctx.buffer[have + i] = inp[i]
            RMD160Transform(ctx.state, ctx.buffer)
            off = need
            have = 0
        while off + 64 <= inplen:
            RMD160Transform(ctx.state, inp[off:])
            off += 64
    if off < inplen:
        for i in range(inplen - off):
            ctx.buffer[have + i] = inp[off + i]

def RMD160Final(ctx):
    if False:
        return 10
    size = struct.pack('<Q', ctx.count)
    padlen = 64 - ctx.count // 8 % 64
    if padlen < 1 + 8:
        padlen += 64
    RMD160Update(ctx, PADDING, padlen - 8)
    RMD160Update(ctx, size, 8)
    return struct.pack('<5L', *ctx.state)
assert '37f332f68db77bd9d7edd4969571ad671cf9dd3b' == new('The quick brown fox jumps over the lazy dog').hexdigest()
assert '132072df690933835eb8b6ad0b77e7b6f14acad7' == new('The quick brown fox jumps over the lazy cog').hexdigest()
assert '9c1185a5c5e9fc54612808977ee8f548b2258d31' == new('').hexdigest()