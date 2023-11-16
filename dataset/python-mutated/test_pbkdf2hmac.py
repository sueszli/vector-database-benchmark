import pytest
from cryptography.exceptions import AlreadyFinalized, InvalidKey, _Reasons
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from ...doubles import DummyHashAlgorithm
from ...utils import raises_unsupported_algorithm

class TestPBKDF2HMAC:

    def test_already_finalized(self, backend):
        if False:
            print('Hello World!')
        kdf = PBKDF2HMAC(hashes.SHA1(), 20, b'salt', 10, backend)
        kdf.derive(b'password')
        with pytest.raises(AlreadyFinalized):
            kdf.derive(b'password2')
        kdf = PBKDF2HMAC(hashes.SHA1(), 20, b'salt', 10, backend)
        key = kdf.derive(b'password')
        with pytest.raises(AlreadyFinalized):
            kdf.verify(b'password', key)
        kdf = PBKDF2HMAC(hashes.SHA1(), 20, b'salt', 10, backend)
        kdf.verify(b'password', key)
        with pytest.raises(AlreadyFinalized):
            kdf.verify(b'password', key)

    def test_unsupported_algorithm(self, backend):
        if False:
            i = 10
            return i + 15
        with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_HASH):
            PBKDF2HMAC(DummyHashAlgorithm(), 20, b'salt', 10, backend)

    def test_invalid_key(self, backend):
        if False:
            i = 10
            return i + 15
        kdf = PBKDF2HMAC(hashes.SHA1(), 20, b'salt', 10, backend)
        key = kdf.derive(b'password')
        kdf = PBKDF2HMAC(hashes.SHA1(), 20, b'salt', 10, backend)
        with pytest.raises(InvalidKey):
            kdf.verify(b'password2', key)

    def test_unicode_error_with_salt(self, backend):
        if False:
            return 10
        with pytest.raises(TypeError):
            PBKDF2HMAC(hashes.SHA1(), 20, 'salt', 10, backend)

    def test_unicode_error_with_key_material(self, backend):
        if False:
            i = 10
            return i + 15
        kdf = PBKDF2HMAC(hashes.SHA1(), 20, b'salt', 10, backend)
        with pytest.raises(TypeError):
            kdf.derive('unicode here')

    def test_buffer_protocol(self, backend):
        if False:
            while True:
                i = 10
        kdf = PBKDF2HMAC(hashes.SHA1(), 10, b'salt', 10, backend)
        data = bytearray(b'data')
        assert kdf.derive(data) == b'\xe9n\xaa\x81\xbbt\xa4\xf6\x08\xce'