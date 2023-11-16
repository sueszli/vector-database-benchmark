"""Pure-Python RSA implementation."""
import os
import math
import hashlib

def SHA1(x):
    if False:
        print('Hello World!')
    return hashlib.sha1(x).digest()
import zlib
length = len(zlib.compress(os.urandom(1000)))
assert length > 900

def getRandomBytes(howMany):
    if False:
        print('Hello World!')
    b = bytearray(os.urandom(howMany))
    assert len(b) == howMany
    return b
prngName = 'os.urandom'

def bytesToNumber(b):
    if False:
        while True:
            i = 10
    total = 0
    multiplier = 1
    for count in range(len(b) - 1, -1, -1):
        byte = b[count]
        total += multiplier * byte
        multiplier *= 256
    return total

def numberToByteArray(n, howManyBytes=None):
    if False:
        print('Hello World!')
    'Convert an integer into a bytearray, zero-pad to howManyBytes.\n\n    The returned bytearray may be smaller than howManyBytes, but will\n    not be larger.  The returned bytearray will contain a big-endian\n    encoding of the input integer (n).\n    '
    if howManyBytes is None:
        howManyBytes = numBytes(n)
    b = bytearray(howManyBytes)
    for count in range(howManyBytes - 1, -1, -1):
        b[count] = int(n % 256)
        n >>= 8
    return b

def mpiToNumber(mpi):
    if False:
        print('Hello World!')
    if ord(mpi[4]) & 128 != 0:
        raise AssertionError()
    b = bytearray(mpi[4:])
    return bytesToNumber(b)

def numberToMPI(n):
    if False:
        while True:
            i = 10
    b = numberToByteArray(n)
    ext = 0
    if numBits(n) & 7 == 0:
        ext = 1
    length = numBytes(n) + ext
    b = bytearray(4 + ext) + b
    b[0] = length >> 24 & 255
    b[1] = length >> 16 & 255
    b[2] = length >> 8 & 255
    b[3] = length & 255
    return bytes(b)

def numBits(n):
    if False:
        i = 10
        return i + 15
    if n == 0:
        return 0
    s = '%x' % n
    return (len(s) - 1) * 4 + {'0': 0, '1': 1, '2': 2, '3': 2, '4': 3, '5': 3, '6': 3, '7': 3, '8': 4, '9': 4, 'a': 4, 'b': 4, 'c': 4, 'd': 4, 'e': 4, 'f': 4}[s[0]]

def numBytes(n):
    if False:
        return 10
    if n == 0:
        return 0
    bits = numBits(n)
    return int(math.ceil(bits / 8.0))

def getRandomNumber(low, high):
    if False:
        return 10
    if low >= high:
        raise AssertionError()
    howManyBits = numBits(high)
    howManyBytes = numBytes(high)
    lastBits = howManyBits % 8
    while 1:
        bytes = getRandomBytes(howManyBytes)
        if lastBits:
            bytes[0] = bytes[0] % (1 << lastBits)
        n = bytesToNumber(bytes)
        if n >= low and n < high:
            return n

def gcd(a, b):
    if False:
        for i in range(10):
            print('nop')
    (a, b) = (max(a, b), min(a, b))
    while b:
        (a, b) = (b, a % b)
    return a

def lcm(a, b):
    if False:
        while True:
            i = 10
    return a * b // gcd(a, b)

def invMod(a, b):
    if False:
        for i in range(10):
            print('nop')
    (c, d) = (a, b)
    (uc, ud) = (1, 0)
    while c != 0:
        q = d // c
        (c, d) = (d - q * c, c)
        (uc, ud) = (ud - q * uc, uc)
    if d == 1:
        return ud % b
    return 0

def powMod(base, power, modulus):
    if False:
        while True:
            i = 10
    if power < 0:
        result = pow(base, power * -1, modulus)
        result = invMod(result, modulus)
        return result
    else:
        return pow(base, power, modulus)

def makeSieve(n):
    if False:
        return 10
    sieve = list(range(n))
    for count in range(2, int(math.sqrt(n)) + 1):
        if sieve[count] == 0:
            continue
        x = sieve[count] * 2
        while x < len(sieve):
            sieve[x] = 0
            x += sieve[count]
    sieve = [x for x in sieve[2:] if x]
    return sieve
sieve = makeSieve(1000)

def isPrime(n, iterations=5, display=False):
    if False:
        print('Hello World!')
    for x in sieve:
        if x >= n:
            return True
        if n % x == 0:
            return False
    if display:
        print('*', end=' ')
    (s, t) = (n - 1, 0)
    while s % 2 == 0:
        (s, t) = (s // 2, t + 1)
    a = 2
    for count in range(iterations):
        v = powMod(a, s, n)
        if v == 1:
            continue
        i = 0
        while v != n - 1:
            if i == t - 1:
                return False
            else:
                (v, i) = (powMod(v, 2, n), i + 1)
        a = getRandomNumber(2, n)
    return True

def getRandomPrime(bits, display=False):
    if False:
        i = 10
        return i + 15
    if bits < 10:
        raise AssertionError()
    low = 2 ** (bits - 1) * 3 // 2
    high = 2 ** bits - 30
    p = getRandomNumber(low, high)
    p += 29 - p % 30
    while 1:
        if display:
            print('.', end=' ')
        p += 30
        if p >= high:
            p = getRandomNumber(low, high)
            p += 29 - p % 30
        if isPrime(p, display=display):
            return p

def getRandomSafePrime(bits, display=False):
    if False:
        i = 10
        return i + 15
    if bits < 10:
        raise AssertionError()
    low = 2 ** (bits - 2) * 3 // 2
    high = 2 ** (bits - 1) - 30
    q = getRandomNumber(low, high)
    q += 29 - q % 30
    while 1:
        if display:
            print('.', end=' ')
        q += 30
        if q >= high:
            q = getRandomNumber(low, high)
            q += 29 - q % 30
        if isPrime(q, 0, display=display):
            p = 2 * q + 1
            if isPrime(p, display=display):
                if isPrime(q, display=display):
                    return p

class RSAKey(object):

    def __init__(self, n=0, e=0, d=0, p=0, q=0, dP=0, dQ=0, qInv=0):
        if False:
            while True:
                i = 10
        if n and (not e) or (e and (not n)):
            raise AssertionError()
        self.n = n
        self.e = e
        self.d = d
        self.p = p
        self.q = q
        self.dP = dP
        self.dQ = dQ
        self.qInv = qInv
        self.blinder = 0
        self.unblinder = 0

    def __len__(self):
        if False:
            while True:
                i = 10
        'Return the length of this key in bits.\n\n        @rtype: int\n        '
        return numBits(self.n)

    def hasPrivateKey(self):
        if False:
            while True:
                i = 10
        return self.d != 0

    def hashAndSign(self, bytes):
        if False:
            while True:
                i = 10
        'Hash and sign the passed-in bytes.\n\n        This requires the key to have a private component.  It performs\n        a PKCS1-SHA1 signature on the passed-in data.\n\n        @type bytes: str or L{bytearray} of unsigned bytes\n        @param bytes: The value which will be hashed and signed.\n\n        @rtype: L{bytearray} of unsigned bytes.\n        @return: A PKCS1-SHA1 signature on the passed-in data.\n        '
        hashBytes = SHA1(bytearray(bytes))
        prefixedHashBytes = self._addPKCS1SHA1Prefix(hashBytes)
        sigBytes = self.sign(prefixedHashBytes)
        return sigBytes

    def hashAndVerify(self, sigBytes, bytes):
        if False:
            return 10
        'Hash and verify the passed-in bytes with the signature.\n\n        This verifies a PKCS1-SHA1 signature on the passed-in data.\n\n        @type sigBytes: L{bytearray} of unsigned bytes\n        @param sigBytes: A PKCS1-SHA1 signature.\n\n        @type bytes: str or L{bytearray} of unsigned bytes\n        @param bytes: The value which will be hashed and verified.\n\n        @rtype: bool\n        @return: Whether the signature matches the passed-in data.\n        '
        hashBytes = SHA1(bytearray(bytes))
        prefixedHashBytes1 = self._addPKCS1SHA1Prefix(hashBytes, False)
        prefixedHashBytes2 = self._addPKCS1SHA1Prefix(hashBytes, True)
        result1 = self.verify(sigBytes, prefixedHashBytes1)
        result2 = self.verify(sigBytes, prefixedHashBytes2)
        return result1 or result2

    def sign(self, bytes):
        if False:
            return 10
        'Sign the passed-in bytes.\n\n        This requires the key to have a private component.  It performs\n        a PKCS1 signature on the passed-in data.\n\n        @type bytes: L{bytearray} of unsigned bytes\n        @param bytes: The value which will be signed.\n\n        @rtype: L{bytearray} of unsigned bytes.\n        @return: A PKCS1 signature on the passed-in data.\n        '
        if not self.hasPrivateKey():
            raise AssertionError()
        paddedBytes = self._addPKCS1Padding(bytes, 1)
        m = bytesToNumber(paddedBytes)
        if m >= self.n:
            raise ValueError()
        c = self._rawPrivateKeyOp(m)
        sigBytes = numberToByteArray(c, numBytes(self.n))
        return sigBytes

    def verify(self, sigBytes, bytes):
        if False:
            i = 10
            return i + 15
        'Verify the passed-in bytes with the signature.\n\n        This verifies a PKCS1 signature on the passed-in data.\n\n        @type sigBytes: L{bytearray} of unsigned bytes\n        @param sigBytes: A PKCS1 signature.\n\n        @type bytes: L{bytearray} of unsigned bytes\n        @param bytes: The value which will be verified.\n\n        @rtype: bool\n        @return: Whether the signature matches the passed-in data.\n        '
        if len(sigBytes) != numBytes(self.n):
            return False
        paddedBytes = self._addPKCS1Padding(bytes, 1)
        c = bytesToNumber(sigBytes)
        if c >= self.n:
            return False
        m = self._rawPublicKeyOp(c)
        checkBytes = numberToByteArray(m, numBytes(self.n))
        return checkBytes == paddedBytes

    def encrypt(self, bytes):
        if False:
            while True:
                i = 10
        'Encrypt the passed-in bytes.\n\n        This performs PKCS1 encryption of the passed-in data.\n\n        @type bytes: L{bytearray} of unsigned bytes\n        @param bytes: The value which will be encrypted.\n\n        @rtype: L{bytearray} of unsigned bytes.\n        @return: A PKCS1 encryption of the passed-in data.\n        '
        paddedBytes = self._addPKCS1Padding(bytes, 2)
        m = bytesToNumber(paddedBytes)
        if m >= self.n:
            raise ValueError()
        c = self._rawPublicKeyOp(m)
        encBytes = numberToByteArray(c, numBytes(self.n))
        return encBytes

    def decrypt(self, encBytes):
        if False:
            for i in range(10):
                print('nop')
        'Decrypt the passed-in bytes.\n\n        This requires the key to have a private component.  It performs\n        PKCS1 decryption of the passed-in data.\n\n        @type encBytes: L{bytearray} of unsigned bytes\n        @param encBytes: The value which will be decrypted.\n\n        @rtype: L{bytearray} of unsigned bytes or None.\n        @return: A PKCS1 decryption of the passed-in data or None if\n        the data is not properly formatted.\n        '
        if not self.hasPrivateKey():
            raise AssertionError()
        if len(encBytes) != numBytes(self.n):
            return None
        c = bytesToNumber(encBytes)
        if c >= self.n:
            return None
        m = self._rawPrivateKeyOp(c)
        decBytes = numberToByteArray(m, numBytes(self.n))
        if decBytes[0] != 0 or decBytes[1] != 2:
            return None
        for x in range(1, len(decBytes) - 1):
            if decBytes[x] == 0:
                break
        else:
            return None
        return decBytes[x + 1:]

    def _addPKCS1SHA1Prefix(self, bytes, withNULL=True):
        if False:
            for i in range(10):
                print('nop')
        if not withNULL:
            prefixBytes = bytearray([48, 31, 48, 7, 6, 5, 43, 14, 3, 2, 26, 4, 20])
        else:
            prefixBytes = bytearray([48, 33, 48, 9, 6, 5, 43, 14, 3, 2, 26, 5, 0, 4, 20])
        prefixedBytes = prefixBytes + bytes
        return prefixedBytes

    def _addPKCS1Padding(self, bytes, blockType):
        if False:
            for i in range(10):
                print('nop')
        padLength = numBytes(self.n) - (len(bytes) + 3)
        if blockType == 1:
            pad = [255] * padLength
        elif blockType == 2:
            pad = bytearray(0)
            while len(pad) < padLength:
                padBytes = getRandomBytes(padLength * 2)
                pad = [b for b in padBytes if b != 0]
                pad = pad[:padLength]
        else:
            raise AssertionError()
        padding = bytearray([0, blockType] + pad + [0])
        paddedBytes = padding + bytes
        return paddedBytes

    def _rawPrivateKeyOp(self, m):
        if False:
            while True:
                i = 10
        if not self.blinder:
            self.unblinder = getRandomNumber(2, self.n)
            self.blinder = powMod(invMod(self.unblinder, self.n), self.e, self.n)
        m = m * self.blinder % self.n
        c = self._rawPrivateKeyOpHelper(m)
        c = c * self.unblinder % self.n
        self.blinder = self.blinder * self.blinder % self.n
        self.unblinder = self.unblinder * self.unblinder % self.n
        return c

    def _rawPrivateKeyOpHelper(self, m):
        if False:
            return 10
        s1 = powMod(m, self.dP, self.p)
        s2 = powMod(m, self.dQ, self.q)
        h = (s1 - s2) * self.qInv % self.p
        c = s2 + self.q * h
        return c

    def _rawPublicKeyOp(self, c):
        if False:
            for i in range(10):
                print('nop')
        m = powMod(c, self.e, self.n)
        return m

    def acceptsPassword(self):
        if False:
            i = 10
            return i + 15
        return False

    def generate(bits):
        if False:
            while True:
                i = 10
        key = RSAKey()
        p = getRandomPrime(bits // 2, False)
        q = getRandomPrime(bits // 2, False)
        t = lcm(p - 1, q - 1)
        key.n = p * q
        key.e = 65537
        key.d = invMod(key.e, t)
        key.p = p
        key.q = q
        key.dP = key.d % (p - 1)
        key.dQ = key.d % (q - 1)
        key.qInv = invMod(q, p)
        return key
    generate = staticmethod(generate)