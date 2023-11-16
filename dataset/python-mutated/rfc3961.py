"""
Implementation of RFC 3961's cryptographic functions
"""
import math
import os
import struct
from scapy.compat import orb, chb, int_bytes, bytes_int, plain_str
try:
    try:
        from Cryptodome.Cipher import AES, DES3, ARC4, DES
        from Cryptodome.Hash import HMAC, MD4, MD5, SHA
        from Cryptodome.Protocol.KDF import PBKDF2
    except ImportError:
        from Crypto.Cipher import AES, DES3, ARC4, DES
        from Crypto.Hash import HMAC, MD4, MD5, SHA
        from Crypto.Protocol.KDF import PBKDF2
except ImportError:
    raise ImportError('To use kerberos cryptography, you need to install pycryptodome.\npip install pycryptodome')
__all__ = ['EncryptionType', 'ChecksumType', 'Key', 'InvalidChecksum']

class EncryptionType:
    DES_CRC = 1
    DES_MD4 = 2
    DES_MD5 = 3
    DES3 = 16
    AES128 = 17
    AES256 = 18
    RC4 = 23

class ChecksumType:
    CRC32 = 1
    MD4 = 2
    MD4_DES = 3
    MD5 = 7
    MD5_DES = 8
    SHA1 = 9
    SHA1_DES3 = 12
    SHA1_AES128 = 15
    SHA1_AES256 = 16
    HMAC_MD5 = -138

class InvalidChecksum(ValueError):
    pass

def _n_fold(s, n):
    if False:
        while True:
            i = 10
    '\n    https://www.gnu.org/software/shishi/ides.pdf - APPENDIX B\n    '

    def rot13(x, nb):
        if False:
            return 10
        x = bytes_int(x)
        mod = (1 << nb * 8) - 1
        if nb == 0:
            return x
        elif nb == 1:
            return int_bytes((x >> 5 | x << nb * 8 - 5) & mod, nb)
        else:
            return int_bytes((x >> 13 | x << nb * 8 - 13) & mod, nb)

    def ocadd(x, y, nb):
        if False:
            print('Hello World!')
        v = [a + b for (a, b) in zip(x, y)]
        while any((x & ~255 for x in v)):
            v = [(v[i - nb + 1] >> 8) + (v[i] & 255) for i in range(nb)]
        return bytearray((x for x in v))
    m = len(s)
    lcm = math.lcm(n, m)
    buf = bytearray()
    for _ in range(lcm // m):
        buf += s
        s = rot13(s, m)
    out = b'\x00' * n
    for i in range(0, lcm, n):
        out = ocadd(out, buf[i:i + n], n)
    return bytes(out)

def _zeropad(s, padsize):
    if False:
        return 10
    '\n    Return s padded with 0 bytes to a multiple of padsize.\n    '
    return s + b'\x00' * (-len(s) % padsize)

def _xorbytes(b1, b2):
    if False:
        i = 10
        return i + 15
    '\n    xor two strings together and return the resulting string\n    '
    assert len(b1) == len(b2)
    return bytearray((x ^ y for (x, y) in zip(b1, b2)))

def _mac_equal(mac1, mac2):
    if False:
        for i in range(10):
            print('nop')
    assert len(mac1) == len(mac2)
    res = 0
    for (x, y) in zip(mac1, mac2):
        res |= x ^ y
    return res == 0
WEAK_DES_KEYS = set([b'\x01' * 8, b'\xfe' * 8, b'\xe0' * 4 + b'\xf1' * 4, b'\x1f' * 4 + b'\x0e' * 4, b'\x01\x1f\x01\x1f\x01\x0e\x01\x0e', b'\x1f\x01\x1f\x01\x0e\x01\x0e\x01', b'\x01\xe0\x01\xe0\x01\xf1\x01\xf1', b'\xe0\x01\xe0\x01\xf1\x01\xf1\x01', b'\x01\xfe\x01\xfe\x01\xfe\x01\xfe', b'\xfe\x01\xfe\x01\xfe\x01\xfe\x01', b'\x1f\xe0\x1f\xe0\x0e\xf1\x0e\xf1', b'\xe0\x1f\xe0\x1f\xf1\x0e\xf1\x0e', b'\x1f\xfe\x1f\xfe\x0e\xfe\x0e\xfe', b'\xfe\x1f\xfe\x1f\xfe\x0e\xfe\x0e', b'\xe0\xfe\xe0\xfe\xf1\xfe\xf1\xfe', b'\xfe\xe0\xfe\xe0\xfe\xf1\xfe\xf1'])

class _EncryptionAlgorithmProfile(object):
    """
    Base class for etype profiles.

    Usable etype classes must define:
    :attr etype: etype number
    :attr keysize: protocol size of key in bytes
    :attr seedsize: random_to_key input size in bytes
    :attr random_to_key: (if the keyspace is not dense)
    :attr string_to_key:
    :attr encrypt:
    :attr decrypt:
    :attr prf:
    """

    @classmethod
    def random_to_key(cls, seed):
        if False:
            while True:
                i = 10
        if len(seed) != cls.seedsize:
            raise ValueError('Wrong seed length')
        return Key(cls.etype, key=seed)

class _SimplifiedEncryptionProfile(_EncryptionAlgorithmProfile):
    """
    Base class for etypes using the RFC 3961 simplified profile.
    Defines the encrypt, decrypt, and prf methods.

    Subclasses must define:

    :param blocksize: Underlying cipher block size in bytes
    :param padsize: Underlying cipher padding multiple (1 or blocksize)
    :param macsize: Size of integrity MAC in bytes
    :param hash: underlying hash function
    :param basic_encrypt, basic_decrypt: Underlying CBC/CTS cipher
    """

    @classmethod
    def derive(cls, key, constant):
        if False:
            for i in range(10):
                print('nop')
        '\n        Also known as "DK" in RFC3961.\n        '
        plaintext = _n_fold(constant, cls.blocksize)
        rndseed = b''
        while len(rndseed) < cls.seedsize:
            ciphertext = cls.basic_encrypt(key, plaintext)
            rndseed += ciphertext
            plaintext = ciphertext
        return cls.random_to_key(rndseed[0:cls.seedsize])

    @classmethod
    def encrypt(cls, key, keyusage, plaintext, confounder):
        if False:
            while True:
                i = 10
        '\n        encryption function\n        '
        ki = cls.derive(key, struct.pack('>IB', keyusage, 85))
        ke = cls.derive(key, struct.pack('>IB', keyusage, 170))
        if confounder is None:
            confounder = os.urandom(cls.blocksize)
        basic_plaintext = confounder + _zeropad(plaintext, cls.padsize)
        hmac = HMAC.new(ki.key, basic_plaintext, cls.hashmod).digest()
        return cls.basic_encrypt(ke, basic_plaintext) + hmac[:cls.macsize]

    @classmethod
    def decrypt(cls, key, keyusage, ciphertext):
        if False:
            for i in range(10):
                print('nop')
        '\n        decryption function\n        '
        ki = cls.derive(key, struct.pack('>IB', keyusage, 85))
        ke = cls.derive(key, struct.pack('>IB', keyusage, 170))
        if len(ciphertext) < cls.blocksize + cls.macsize:
            raise ValueError('Ciphertext too short')
        (basic_ctext, mac) = (bytearray(ciphertext[:-cls.macsize]), bytearray(ciphertext[-cls.macsize:]))
        if len(basic_ctext) % cls.padsize != 0:
            raise ValueError('ciphertext does not meet padding requirement')
        basic_plaintext = cls.basic_decrypt(ke, bytes(basic_ctext))
        hmac = bytearray(HMAC.new(ki.key, basic_plaintext, cls.hashmod).digest())
        expmac = hmac[:cls.macsize]
        if not _mac_equal(mac, expmac):
            raise ValueError('ciphertext integrity failure')
        return bytes(basic_plaintext[cls.blocksize:])

    @classmethod
    def prf(cls, key, string):
        if False:
            while True:
                i = 10
        '\n        pseudo-random function\n        '
        hashval = cls.hashmod.new(string).digest()
        if len(hashval) % cls.blocksize:
            hashval = hashval[:-(len(hashval) % cls.blocksize)]
        kp = cls.derive(key, b'prf')
        return cls.basic_encrypt(kp, hashval)

class _DESCBC(_SimplifiedEncryptionProfile):
    keysize = 8
    seedsize = 8
    blocksize = 8
    padsize = 8
    macsize = 16
    hashmod = MD5

    @classmethod
    def encrypt(cls, key, keyusage, plaintext, confounder):
        if False:
            while True:
                i = 10
        if confounder is None:
            confounder = os.urandom(cls.blocksize)
        basic_plaintext = confounder + b'\x00' * cls.macsize + _zeropad(plaintext, cls.padsize)
        checksum = cls.hashmod.new(basic_plaintext).digest()
        basic_plaintext = basic_plaintext[:len(confounder)] + checksum + basic_plaintext[len(confounder) + len(checksum):]
        return cls.basic_encrypt(key, basic_plaintext)

    @classmethod
    def decrypt(cls, key, keyusage, ciphertext):
        if False:
            return 10
        if len(ciphertext) < cls.blocksize + cls.macsize:
            raise ValueError('ciphertext too short')
        complex_plaintext = cls.basic_decrypt(key, ciphertext)
        cofounder = complex_plaintext[:cls.padsize]
        mac = complex_plaintext[cls.padsize:cls.padsize + cls.macsize]
        message = complex_plaintext[cls.padsize + cls.macsize:]
        expmac = bytearray(cls.hashmod.new(cofounder + b'\x00' * cls.macsize + message).digest())
        if not _mac_equal(mac, expmac):
            raise InvalidChecksum('ciphertext integrity failure')
        return bytes(message)

    @classmethod
    def mit_des_string_to_key(cls, string, salt):
        if False:
            return 10

        def fixparity(deskey):
            if False:
                return 10
            temp = b''
            for i in range(len(deskey)):
                t = bin(orb(deskey[i]))[2:].rjust(8, '0')
                if t[:7].count('1') % 2 == 0:
                    temp += chb(int(t[:7] + '1', 2))
                else:
                    temp += chb(int(t[:7] + '0', 2))
            return temp

        def addparity(l1):
            if False:
                while True:
                    i = 10
            temp = list()
            for byte in l1:
                if bin(byte).count('1') % 2 == 0:
                    byte = byte << 1 | 1
                else:
                    byte = byte << 1 & 254
                temp.append(byte)
            return temp

        def XOR(l1, l2):
            if False:
                while True:
                    i = 10
            temp = list()
            for (b1, b2) in zip(l1, l2):
                temp.append((b1 ^ b2) & 127)
            return temp
        odd = True
        tempstring = [0, 0, 0, 0, 0, 0, 0, 0]
        s = _zeropad(string + salt, cls.padsize)
        for block in [s[i:i + 8] for i in range(0, len(s), 8)]:
            temp56 = list()
            for byte in block:
                temp56.append(orb(byte) & 127)
            if odd is False:
                bintemp = b''
                for byte in temp56:
                    bintemp += bin(byte)[2:].rjust(7, '0').encode()
                bintemp = bintemp[::-1]
                temp56 = list()
                for bits7 in [bintemp[i:i + 7] for i in range(0, len(bintemp), 7)]:
                    temp56.append(int(bits7, 2))
            odd = not odd
            tempstring = XOR(tempstring, temp56)
        tempkey = bytearray(b''.join((chb(byte) for byte in addparity(tempstring))))
        if bytes(tempkey) in WEAK_DES_KEYS:
            tempkey[7] = tempkey[7] ^ 240
        cipher = DES.new(tempkey, DES.MODE_CBC, tempkey)
        chekcsumkey = cipher.encrypt(s)[-8:]
        chekcsumkey = bytearray(fixparity(chekcsumkey))
        if bytes(chekcsumkey) in WEAK_DES_KEYS:
            chekcsumkey[7] = chekcsumkey[7] ^ 240
        return Key(cls.etype, key=bytes(chekcsumkey))

    @classmethod
    def basic_encrypt(cls, key, plaintext):
        if False:
            i = 10
            return i + 15
        assert len(plaintext) % 8 == 0
        des = DES.new(key.key, DES.MODE_CBC, b'\x00' * 8)
        return des.encrypt(bytes(plaintext))

    @classmethod
    def basic_decrypt(cls, key, ciphertext):
        if False:
            for i in range(10):
                print('nop')
        assert len(ciphertext) % 8 == 0
        des = DES.new(key.key, DES.MODE_CBC, b'\x00' * 8)
        return des.decrypt(bytes(ciphertext))

    @classmethod
    def string_to_key(cls, string, salt, params):
        if False:
            return 10
        if params is not None and params != b'':
            raise ValueError('Invalid DES string-to-key parameters')
        key = cls.mit_des_string_to_key(string, salt)
        return key

class _DESMD5(_DESCBC):
    etype = EncryptionType.DES_MD5
    hashmod = MD5

class _DESMD4(_DESCBC):
    etype = EncryptionType.DES_MD4
    hashmod = MD4

class _DES3CBC(_SimplifiedEncryptionProfile):
    etype = EncryptionType.DES3
    keysize = 24
    seedsize = 21
    blocksize = 8
    padsize = 8
    macsize = 20
    hashmod = SHA

    @classmethod
    def random_to_key(cls, seed):
        if False:
            i = 10
            return i + 15

        def expand(seed):
            if False:
                while True:
                    i = 10

            def parity(b):
                if False:
                    while True:
                        i = 10
                b &= ~1
                return b if bin(b & ~1).count('1') % 2 else b | 1
            assert len(seed) == 7
            firstbytes = [parity(b & ~1) for b in seed]
            lastbyte = parity(sum(((seed[i] & 1) << i + 1 for i in range(7))))
            keybytes = bytes(bytearray(firstbytes + [lastbyte]))
            if keybytes in WEAK_DES_KEYS:
                keybytes[7] = keybytes[7] ^ 240
            return bytes(keybytes)
        seed = bytearray(seed)
        if len(seed) != 21:
            raise ValueError('Wrong seed length')
        (k1, k2, k3) = (expand(seed[:7]), expand(seed[7:14]), expand(seed[14:]))
        return Key(cls.etype, key=k1 + k2 + k3)

    @classmethod
    def string_to_key(cls, string, salt, params):
        if False:
            for i in range(10):
                print('nop')
        if params is not None and params != b'':
            raise ValueError('Invalid DES3 string-to-key parameters')
        k = cls.random_to_key(_n_fold(string + salt, 21))
        return cls.derive(k, b'kerberos')

    @classmethod
    def basic_encrypt(cls, key, plaintext):
        if False:
            for i in range(10):
                print('nop')
        assert len(plaintext) % 8 == 0
        des3 = DES3.new(key.key, AES.MODE_CBC, b'\x00' * 8)
        return des3.encrypt(bytes(plaintext))

    @classmethod
    def basic_decrypt(cls, key, ciphertext):
        if False:
            i = 10
            return i + 15
        assert len(ciphertext) % 8 == 0
        des3 = DES3.new(key.key, AES.MODE_CBC, b'\x00' * 8)
        return des3.decrypt(bytes(ciphertext))

class _AESEncryptionType(_SimplifiedEncryptionProfile):
    blocksize = 16
    padsize = 1
    macsize = 12
    hashmod = SHA

    @classmethod
    def string_to_key(cls, string, salt, params):
        if False:
            return 10
        iterations = struct.unpack('>L', params or b'\x00\x00\x10\x00')[0]
        prf = lambda p, s: HMAC.new(p, s, SHA).digest()
        seed = PBKDF2(string, salt, cls.seedsize, iterations, prf)
        tkey = cls.random_to_key(seed)
        return cls.derive(tkey, b'kerberos')

    @classmethod
    def basic_encrypt(cls, key, plaintext):
        if False:
            i = 10
            return i + 15
        assert len(plaintext) >= 16
        aes = AES.new(key.key, AES.MODE_CBC, b'\x00' * 16)
        ctext = aes.encrypt(_zeropad(bytes(plaintext), 16))
        if len(plaintext) > 16:
            lastlen = len(plaintext) % 16 or 16
            ctext = ctext[:-32] + ctext[-16:] + ctext[-32:-16][:lastlen]
        return ctext

    @classmethod
    def basic_decrypt(cls, key, ciphertext):
        if False:
            return 10
        assert len(ciphertext) >= 16
        aes = AES.new(key.key, AES.MODE_ECB)
        if len(ciphertext) == 16:
            return aes.decrypt(ciphertext)
        cblocks = [bytearray(ciphertext[p:p + 16]) for p in range(0, len(ciphertext), 16)]
        lastlen = len(cblocks[-1])
        prev_cblock = bytearray(16)
        plaintext = b''
        for bb in cblocks[:-2]:
            plaintext += _xorbytes(bytearray(aes.decrypt(bytes(bb))), prev_cblock)
            prev_cblock = bb
        bb = bytearray(aes.decrypt(bytes(cblocks[-2])))
        lastplaintext = _xorbytes(bb[:lastlen], cblocks[-1])
        omitted = bb[lastlen:]
        plaintext += _xorbytes(bytearray(aes.decrypt(bytes(cblocks[-1]) + bytes(omitted))), prev_cblock)
        return plaintext + lastplaintext

class _AES128CTS(_AESEncryptionType):
    etype = 17
    keysize = 16
    seedsize = 16

class _AES256CTS(_AESEncryptionType):
    etype = 18
    keysize = 32
    seedsize = 32

class _RC4(_EncryptionAlgorithmProfile):
    etype = 23
    keysize = 16
    seedsize = 16

    @staticmethod
    def usage_str(keyusage):
        if False:
            i = 10
            return i + 15
        table = {3: 8, 23: 13}
        msusage = table[keyusage] if keyusage in table else keyusage
        return struct.pack('<I', msusage)

    @classmethod
    def string_to_key(cls, string, salt, params):
        if False:
            while True:
                i = 10
        utf16string = plain_str(string).encode('UTF-16LE')
        return Key(cls.etype, key=MD4.new(utf16string).digest())

    @classmethod
    def encrypt(cls, key, keyusage, plaintext, confounder):
        if False:
            for i in range(10):
                print('nop')
        if confounder is None:
            confounder = os.urandom(8)
        ki = HMAC.new(key.key, cls.usage_str(keyusage), MD5).digest()
        cksum = HMAC.new(ki, confounder + plaintext, MD5).digest()
        ke = HMAC.new(ki, cksum, MD5).digest()
        return cksum + ARC4.new(ke).encrypt(bytes(confounder + plaintext))

    @classmethod
    def decrypt(cls, key, keyusage, ciphertext):
        if False:
            for i in range(10):
                print('nop')
        if len(ciphertext) < 24:
            raise ValueError('ciphertext too short')
        (cksum, basic_ctext) = (bytearray(ciphertext[:16]), bytearray(ciphertext[16:]))
        ki = HMAC.new(key.key, cls.usage_str(keyusage), MD5).digest()
        ke = HMAC.new(ki, cksum, MD5).digest()
        basic_plaintext = bytearray(ARC4.new(ke).decrypt(bytes(basic_ctext)))
        exp_cksum = bytearray(HMAC.new(ki, basic_plaintext, MD5).digest())
        ok = _mac_equal(cksum, exp_cksum)
        if not ok and keyusage == 9:
            ki = HMAC.new(key.key, struct.pack('<I', 8), MD5).digest()
            exp_cksum = HMAC.new(ki, basic_plaintext, MD5).digest()
            ok = _mac_equal(cksum, exp_cksum)
        if not ok:
            raise InvalidChecksum('ciphertext integrity failure')
        return bytes(basic_plaintext[8:])

    @classmethod
    def prf(cls, key, string):
        if False:
            while True:
                i = 10
        return HMAC.new(key.key, bytes(string), SHA).digest()

class _ChecksumProfile(object):

    @classmethod
    def verify(cls, key, keyusage, text, cksum):
        if False:
            return 10
        expected = cls.checksum(key, keyusage, text)
        if not _mac_equal(bytearray(cksum), bytearray(expected)):
            raise InvalidChecksum('checksum verification failure')

class _SimplifiedChecksum(_ChecksumProfile):

    @classmethod
    def checksum(cls, key, keyusage, text):
        if False:
            while True:
                i = 10
        kc = cls.enc.derive(key, struct.pack('>IB', keyusage, 153))
        hmac = HMAC.new(kc.key, text, cls.enc.hashmod).digest()
        return hmac[:cls.macsize]

    @classmethod
    def verify(cls, key, keyusage, text, cksum):
        if False:
            for i in range(10):
                print('nop')
        if key.etype != cls.enc.etype:
            raise ValueError('Wrong key type for checksum')
        super(_SimplifiedChecksum, cls).verify(key, keyusage, text, cksum)

class _SHA1AES128(_SimplifiedChecksum):
    macsize = 12
    enc = _AES128CTS

class _SHA1AES256(_SimplifiedChecksum):
    macsize = 12
    enc = _AES256CTS

class _SHA1DES3(_SimplifiedChecksum):
    macsize = 20
    enc = _DES3CBC

class _HMACMD5(_ChecksumProfile):

    @classmethod
    def checksum(cls, key, keyusage, text):
        if False:
            print('Hello World!')
        ksign = HMAC.new(key.key, b'signaturekey\x00', MD5).digest()
        md5hash = MD5.new(_RC4.usage_str(keyusage) + text).digest()
        return HMAC.new(ksign, md5hash, MD5).digest()

    @classmethod
    def verify(cls, key, keyusage, text, cksum):
        if False:
            for i in range(10):
                print('nop')
        if key.etype != EncryptionType.RC4:
            raise ValueError('Wrong key type for checksum')
        super(_HMACMD5, cls).verify(key, keyusage, text, cksum)
_enctypes = {EncryptionType.DES_MD5: _DESMD5, EncryptionType.DES_MD4: _DESMD4, EncryptionType.DES3: _DES3CBC, EncryptionType.AES128: _AES128CTS, EncryptionType.AES256: _AES256CTS, EncryptionType.RC4: _RC4}
_checksums = {ChecksumType.SHA1_DES3: _SHA1DES3, ChecksumType.SHA1_AES128: _SHA1AES128, ChecksumType.SHA1_AES256: _SHA1AES256, ChecksumType.HMAC_MD5: _HMACMD5, 4294967158: _HMACMD5}

class Key(object):

    def __init__(self, etype, cksumtype=None, key=None):
        if False:
            i = 10
            return i + 15
        self.eptype = etype
        try:
            self.ep = _enctypes[etype]
        except ValueError:
            raise ValueError("Unknown etype '%s'" % etype)
        self.cksumtype = cksumtype
        if cksumtype is not None:
            try:
                self.cp = _checksums[etype]
            except ValueError:
                raise ValueError("Unknown etype '%s'" % etype)
        if key is not None and len(key) != self.ep.keysize:
            raise ValueError('Wrong key length. Got %s. Expected %s' % (len(key), self.ep.keysize))
        self.key = key

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Key %s%s>' % (self.eptype, ' (%s octets)' % len(self.key) if self.key is not None else '')

    def encrypt(self, keyusage, plaintext, confounder=None):
        if False:
            print('Hello World!')
        return self.ep.encrypt(self, keyusage, bytes(plaintext), confounder)

    def decrypt(self, keyusage, ciphertext):
        if False:
            return 10
        return self.ep.decrypt(self, keyusage, ciphertext)

    def prf(self, string):
        if False:
            return 10
        return self.ep.prf(self, string)

    def make_checksum(self, keyusage, text):
        if False:
            for i in range(10):
                print('nop')
        if self.cksumtype is None:
            raise ValueError('checksumtype not specified !')
        return self.cp.checksum(self, keyusage, text)

    def verify_checksum(self, keyusage, text, cksum):
        if False:
            for i in range(10):
                print('nop')
        if self.cksumtype is None:
            raise ValueError('checksumtype not specified !')
        self.cp.verify(self, keyusage, text, cksum)

    @classmethod
    def random_to_key(cls, etype, seed):
        if False:
            return 10
        try:
            ep = _enctypes[etype]
        except ValueError:
            raise ValueError("Unknown etype '%s'" % etype)
        if len(seed) != ep.seedsize:
            raise ValueError('Wrong crypto seed length')
        return ep.random_to_key(seed)

    @classmethod
    def string_to_key(cls, etype, string, salt, params=None):
        if False:
            i = 10
            return i + 15
        try:
            ep = _enctypes[etype]
        except ValueError:
            raise ValueError("Unknown etype '%s'" % etype)
        return ep.string_to_key(string, salt, params)

def KRB_FX_CF2(key1, key2, pepper1, pepper2):
    if False:
        print('Hello World!')
    '\n    KRB-FX-CF2 RFC6113\n    '

    def prfplus(key, pepper):
        if False:
            print('Hello World!')
        out = b''
        count = 1
        while len(out) < key.ep.seedsize:
            out += key.prf(chb(count) + pepper)
            count += 1
        return out[:key.ep.seedsize]
    return Key(key1.eptype, key=bytes(_xorbytes(bytearray(prfplus(key1, pepper1)), bytearray(prfplus(key2, pepper2)))))