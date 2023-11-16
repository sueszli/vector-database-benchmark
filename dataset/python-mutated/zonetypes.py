"""Common zone-related types."""
import hashlib
import dns.enum

class DigestScheme(dns.enum.IntEnum):
    """ZONEMD Scheme"""
    SIMPLE = 1

    @classmethod
    def _maximum(cls):
        if False:
            return 10
        return 255

class DigestHashAlgorithm(dns.enum.IntEnum):
    """ZONEMD Hash Algorithm"""
    SHA384 = 1
    SHA512 = 2

    @classmethod
    def _maximum(cls):
        if False:
            for i in range(10):
                print('nop')
        return 255
_digest_hashers = {DigestHashAlgorithm.SHA384: hashlib.sha384, DigestHashAlgorithm.SHA512: hashlib.sha512}