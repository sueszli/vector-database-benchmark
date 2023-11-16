"""
Stateless HKDF for TLS 1.3.
"""
import struct
from scapy.config import conf, crypto_validator
from scapy.layers.tls.crypto.pkcs1 import _get_hash
if conf.crypto_valid:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand
    from cryptography.hazmat.primitives.hashes import Hash
    from cryptography.hazmat.primitives.hmac import HMAC

class TLS13_HKDF(object):

    @crypto_validator
    def __init__(self, hash_name='sha256'):
        if False:
            print('Hello World!')
        self.hash = _get_hash(hash_name)

    @crypto_validator
    def extract(self, salt, ikm):
        if False:
            print('Hello World!')
        h = self.hash
        hkdf = HKDF(h, h.digest_size, salt, None, default_backend())
        if ikm is None:
            ikm = b'\x00' * h.digest_size
        return hkdf._extract(ikm)

    @crypto_validator
    def expand(self, prk, info, L):
        if False:
            while True:
                i = 10
        h = self.hash
        hkdf = HKDFExpand(h, L, info, default_backend())
        return hkdf.derive(prk)

    @crypto_validator
    def expand_label(self, secret, label, hash_value, length):
        if False:
            while True:
                i = 10
        hkdf_label = struct.pack('!H', length)
        hkdf_label += struct.pack('B', 6 + len(label))
        hkdf_label += b'tls13 '
        hkdf_label += label
        hkdf_label += struct.pack('B', len(hash_value))
        hkdf_label += hash_value
        return self.expand(secret, hkdf_label, length)

    @crypto_validator
    def derive_secret(self, secret, label, messages):
        if False:
            for i in range(10):
                print('nop')
        h = Hash(self.hash, backend=default_backend())
        h.update(messages)
        hash_messages = h.finalize()
        hash_len = self.hash.digest_size
        return self.expand_label(secret, label, hash_messages, hash_len)

    @crypto_validator
    def compute_verify_data(self, basekey, handshake_context):
        if False:
            return 10
        hash_len = self.hash.digest_size
        finished_key = self.expand_label(basekey, b'finished', b'', hash_len)
        h = Hash(self.hash, backend=default_backend())
        h.update(handshake_context)
        hash_value = h.finalize()
        hm = HMAC(finished_key, self.hash, default_backend())
        hm.update(hash_value)
        return hm.finalize()