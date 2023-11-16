"""
PKCS #1 methods as defined in RFC 3447.

We cannot rely solely on the cryptography library, because the openssl package
used by the cryptography library may not implement the md5-sha1 hash, as with
Ubuntu or OSX. This is why we reluctantly keep some legacy crypto here.
"""
from scapy.compat import bytes_encode, hex_bytes, bytes_hex
from scapy.config import conf, crypto_validator
from scapy.error import warning
if conf.crypto_valid:
    from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.hashes import HashAlgorithm

def pkcs_os2ip(s):
    if False:
        for i in range(10):
            print('nop')
    '\n    OS2IP conversion function from RFC 3447.\n\n    :param s: octet string to be converted\n    :return: n, the corresponding nonnegative integer\n    '
    return int(bytes_hex(s), 16)

def pkcs_i2osp(n, sLen):
    if False:
        return 10
    '\n    I2OSP conversion function from RFC 3447.\n    The length parameter allows the function to perform the padding needed.\n    Note that the user is responsible for providing a sufficient xLen.\n\n    :param n: nonnegative integer to be converted\n    :param sLen: intended length of the resulting octet string\n    :return: corresponding octet string\n    '
    fmt = '%%0%dx' % (2 * sLen)
    return hex_bytes(fmt % n)

def pkcs_ilen(n):
    if False:
        print('Hello World!')
    '\n    This is a log base 256 which determines the minimum octet string\n    length for unequivocal representation of integer n by pkcs_i2osp.\n    '
    i = 0
    while n > 0:
        n >>= 8
        i += 1
    return i

@crypto_validator
def _legacy_pkcs1_v1_5_encode_md5_sha1(M, emLen):
    if False:
        i = 10
        return i + 15
    '\n    Legacy method for PKCS1 v1.5 encoding with MD5-SHA1 hash.\n    '
    M = bytes_encode(M)
    md5_hash = hashes.Hash(_get_hash('md5'), backend=default_backend())
    md5_hash.update(M)
    sha1_hash = hashes.Hash(_get_hash('sha1'), backend=default_backend())
    sha1_hash.update(M)
    H = md5_hash.finalize() + sha1_hash.finalize()
    if emLen < 36 + 11:
        warning('pkcs_emsa_pkcs1_v1_5_encode: intended encoded message length too short')
        return None
    PS = b'\xff' * (emLen - 36 - 3)
    return b'\x00' + b'\x01' + PS + b'\x00' + H
_get_hash = None
if conf.crypto_valid:

    class MD5_SHA1(HashAlgorithm):
        name = 'md5-sha1'
        digest_size = 36
        block_size = 64
    _hashes = {'md5': hashes.MD5, 'sha1': hashes.SHA1, 'sha224': hashes.SHA224, 'sha256': hashes.SHA256, 'sha384': hashes.SHA384, 'sha512': hashes.SHA512, 'md5-sha1': MD5_SHA1}

    def _get_hash(hashStr):
        if False:
            while True:
                i = 10
        try:
            return _hashes[hashStr]()
        except KeyError:
            raise KeyError('Unknown hash function %s' % hashStr)

    def _get_padding(padStr, mgf=padding.MGF1, h=hashes.SHA256, label=None):
        if False:
            i = 10
            return i + 15
        if padStr == 'pkcs':
            return padding.PKCS1v15()
        elif padStr == 'pss':
            return padding.PSS(mgf=mgf(h), salt_length=h.digest_size)
        elif padStr == 'oaep':
            return padding.OAEP(mgf=mgf(h), algorithm=h, label=label)
        else:
            warning('Key.encrypt(): Unknown padding type (%s)', padStr)
            return None

class _EncryptAndVerifyRSA(object):

    @crypto_validator
    def encrypt(self, m, t='pkcs', h='sha256', mgf=None, L=None):
        if False:
            return 10
        mgf = mgf or padding.MGF1
        h = _get_hash(h)
        pad = _get_padding(t, mgf, h, L)
        return self.pubkey.encrypt(m, pad)

    @crypto_validator
    def verify(self, M, S, t='pkcs', h='sha256', mgf=None, L=None):
        if False:
            return 10
        M = bytes_encode(M)
        mgf = mgf or padding.MGF1
        h = _get_hash(h)
        pad = _get_padding(t, mgf, h, L)
        try:
            try:
                self.pubkey.verify(S, M, pad, h)
            except UnsupportedAlgorithm:
                if t != 'pkcs' and h != 'md5-sha1':
                    raise UnsupportedAlgorithm('RSA verification with %s' % h)
                self._legacy_verify_md5_sha1(M, S)
            return True
        except InvalidSignature:
            return False

    def _legacy_verify_md5_sha1(self, M, S):
        if False:
            while True:
                i = 10
        k = self._modulusLen // 8
        if len(S) != k:
            warning('invalid signature (len(S) != k)')
            return False
        s = pkcs_os2ip(S)
        n = self._modulus
        if s > n - 1:
            warning('Key._rsaep() expects a long between 0 and n-1')
            return None
        m = pow(s, self._pubExp, n)
        EM = pkcs_i2osp(m, k)
        EMPrime = _legacy_pkcs1_v1_5_encode_md5_sha1(M, k)
        if EMPrime is None:
            warning('Key._rsassa_pkcs1_v1_5_verify(): unable to encode.')
            return False
        return EM == EMPrime

class _DecryptAndSignRSA(object):

    @crypto_validator
    def decrypt(self, C, t='pkcs', h='sha256', mgf=None, L=None):
        if False:
            return 10
        mgf = mgf or padding.MGF1
        h = _get_hash(h)
        pad = _get_padding(t, mgf, h, L)
        return self.key.decrypt(C, pad)

    @crypto_validator
    def sign(self, M, t='pkcs', h='sha256', mgf=None, L=None):
        if False:
            return 10
        M = bytes_encode(M)
        mgf = mgf or padding.MGF1
        h = _get_hash(h)
        pad = _get_padding(t, mgf, h, L)
        try:
            return self.key.sign(M, pad, h)
        except UnsupportedAlgorithm:
            if t != 'pkcs' and h != 'md5-sha1':
                raise UnsupportedAlgorithm('RSA signature with %s' % h)
            return self._legacy_sign_md5_sha1(M)

    def _legacy_sign_md5_sha1(self, M):
        if False:
            i = 10
            return i + 15
        M = bytes_encode(M)
        k = self._modulusLen // 8
        EM = _legacy_pkcs1_v1_5_encode_md5_sha1(M, k)
        if EM is None:
            warning('Key._rsassa_pkcs1_v1_5_sign(): unable to encode')
            return None
        m = pkcs_os2ip(EM)
        n = self._modulus
        if m > n - 1:
            warning('Key._rsaep() expects a long between 0 and n-1')
            return None
        privExp = self.key.private_numbers().d
        s = pow(m, privExp, n)
        return pkcs_i2osp(s, k)