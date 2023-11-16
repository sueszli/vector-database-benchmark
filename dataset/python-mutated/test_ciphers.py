import binascii
import mmap
import os
import sys
import pytest
from cryptography.exceptions import AlreadyFinalized, _Reasons
from cryptography.hazmat.primitives import ciphers
from cryptography.hazmat.primitives.ciphers import modes
from cryptography.hazmat.primitives.ciphers.algorithms import AES, ARC4, Camellia, TripleDES, _BlowfishInternal, _CAST5Internal, _IDEAInternal, _SEEDInternal
from ...utils import load_nist_vectors, load_vectors_from_file, raises_unsupported_algorithm

class TestAES:

    @pytest.mark.parametrize(('key', 'keysize'), [(b'0' * 32, 128), (b'0' * 48, 192), (b'0' * 64, 256)])
    def test_key_size(self, key, keysize):
        if False:
            i = 10
            return i + 15
        cipher = AES(binascii.unhexlify(key))
        assert cipher.key_size == keysize

    def test_invalid_key_size(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError):
            AES(binascii.unhexlify(b'0' * 12))

    def test_invalid_key_type(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError, match='key must be bytes'):
            AES('0' * 32)

class TestAESXTS:

    @pytest.mark.parametrize('mode', (modes.CBC, modes.CTR, modes.CFB, modes.CFB8, modes.OFB))
    def test_invalid_key_size_with_mode(self, mode, backend):
        if False:
            return 10
        with pytest.raises(ValueError):
            ciphers.Cipher(AES(b'0' * 64), mode(b'0' * 16), backend)

    def test_xts_tweak_not_bytes(self):
        if False:
            print('Hello World!')
        with pytest.raises(TypeError):
            modes.XTS(32)

    def test_xts_tweak_too_small(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            modes.XTS(b'0')

    def test_xts_wrong_key_size(self, backend):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            ciphers.Cipher(AES(b'0' * 16), modes.XTS(b'0' * 16), backend)

class TestGCM:

    @pytest.mark.parametrize('size', [7, 129])
    def test_gcm_min_max(self, size):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            modes.GCM(b'0' * size)

class TestCamellia:

    @pytest.mark.parametrize(('key', 'keysize'), [(b'0' * 32, 128), (b'0' * 48, 192), (b'0' * 64, 256)])
    def test_key_size(self, key, keysize):
        if False:
            while True:
                i = 10
        cipher = Camellia(binascii.unhexlify(key))
        assert cipher.key_size == keysize

    def test_invalid_key_size(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError):
            Camellia(binascii.unhexlify(b'0' * 12))

    def test_invalid_key_type(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError, match='key must be bytes'):
            Camellia('0' * 32)

class TestTripleDES:

    @pytest.mark.parametrize('key', [b'0' * 16, b'0' * 32, b'0' * 48])
    def test_key_size(self, key):
        if False:
            for i in range(10):
                print('nop')
        cipher = TripleDES(binascii.unhexlify(key))
        assert cipher.key_size == 192

    def test_invalid_key_size(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            TripleDES(binascii.unhexlify(b'0' * 12))

    def test_invalid_key_type(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError, match='key must be bytes'):
            TripleDES('0' * 16)

class TestBlowfish:

    @pytest.mark.parametrize(('key', 'keysize'), [(b'0' * (keysize // 4), keysize) for keysize in range(32, 449, 8)])
    def test_key_size(self, key, keysize):
        if False:
            i = 10
            return i + 15
        cipher = _BlowfishInternal(binascii.unhexlify(key))
        assert cipher.key_size == keysize

    def test_invalid_key_size(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            _BlowfishInternal(binascii.unhexlify(b'0' * 6))

    def test_invalid_key_type(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError, match='key must be bytes'):
            _BlowfishInternal('0' * 8)

class TestCAST5:

    @pytest.mark.parametrize(('key', 'keysize'), [(b'0' * (keysize // 4), keysize) for keysize in range(40, 129, 8)])
    def test_key_size(self, key, keysize):
        if False:
            i = 10
            return i + 15
        cipher = _CAST5Internal(binascii.unhexlify(key))
        assert cipher.key_size == keysize

    def test_invalid_key_size(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            _CAST5Internal(binascii.unhexlify(b'0' * 34))

    def test_invalid_key_type(self):
        if False:
            return 10
        with pytest.raises(TypeError, match='key must be bytes'):
            _CAST5Internal('0' * 10)

class TestARC4:

    @pytest.mark.parametrize(('key', 'keysize'), [(b'0' * 10, 40), (b'0' * 14, 56), (b'0' * 16, 64), (b'0' * 20, 80), (b'0' * 32, 128), (b'0' * 48, 192), (b'0' * 64, 256)])
    def test_key_size(self, key, keysize):
        if False:
            while True:
                i = 10
        cipher = ARC4(binascii.unhexlify(key))
        assert cipher.key_size == keysize

    def test_invalid_key_size(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            ARC4(binascii.unhexlify(b'0' * 34))

    def test_invalid_key_type(self):
        if False:
            return 10
        with pytest.raises(TypeError, match='key must be bytes'):
            ARC4('0' * 10)

class TestIDEA:

    def test_key_size(self):
        if False:
            return 10
        cipher = _IDEAInternal(b'\x00' * 16)
        assert cipher.key_size == 128

    def test_invalid_key_size(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            _IDEAInternal(b'\x00' * 17)

    def test_invalid_key_type(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError, match='key must be bytes'):
            _IDEAInternal('0' * 16)

class TestSEED:

    def test_key_size(self):
        if False:
            i = 10
            return i + 15
        cipher = _SEEDInternal(b'\x00' * 16)
        assert cipher.key_size == 128

    def test_invalid_key_size(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            _SEEDInternal(b'\x00' * 17)

    def test_invalid_key_type(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError, match='key must be bytes'):
            _SEEDInternal('0' * 16)

def test_invalid_mode_algorithm():
    if False:
        for i in range(10):
            print('nop')
    with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_CIPHER):
        ciphers.Cipher(ARC4(b'\x00' * 16), modes.GCM(b'\x00' * 12))
    with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_CIPHER):
        ciphers.Cipher(ARC4(b'\x00' * 16), modes.CBC(b'\x00' * 12))
    with raises_unsupported_algorithm(_Reasons.UNSUPPORTED_CIPHER):
        ciphers.Cipher(ARC4(b'\x00' * 16), modes.CTR(b'\x00' * 12))

@pytest.mark.supported(only_if=lambda backend: backend.cipher_supported(AES(b'\x00' * 16), modes.ECB()), skip_message='Does not support AES ECB')
class TestCipherUpdateInto:

    @pytest.mark.parametrize('params', load_vectors_from_file(os.path.join('ciphers', 'AES', 'ECB', 'ECBGFSbox128.rsp'), load_nist_vectors))
    def test_update_into(self, params, backend):
        if False:
            return 10
        key = binascii.unhexlify(params['key'])
        pt = binascii.unhexlify(params['plaintext'])
        ct = binascii.unhexlify(params['ciphertext'])
        c = ciphers.Cipher(AES(key), modes.ECB(), backend)
        encryptor = c.encryptor()
        buf = bytearray(len(pt) + 15)
        res = encryptor.update_into(pt, buf)
        assert res == len(pt)
        assert bytes(buf)[:res] == ct

    @pytest.mark.supported(only_if=lambda backend: backend.cipher_supported(AES(b'\x00' * 16), modes.GCM(b'0' * 12)), skip_message='Does not support AES GCM')
    def test_update_into_gcm(self, backend):
        if False:
            for i in range(10):
                print('nop')
        key = binascii.unhexlify(b'e98b72a9881a84ca6b76e0f43e68647a')
        iv = binascii.unhexlify(b'8b23299fde174053f3d652ba')
        ct = binascii.unhexlify(b'5a3c1cf1985dbb8bed818036fdd5ab42')
        pt = binascii.unhexlify(b'28286a321293253c3e0aa2704a278032')
        c = ciphers.Cipher(AES(key), modes.GCM(iv), backend)
        encryptor = c.encryptor()
        buf = bytearray(len(pt) + 15)
        res = encryptor.update_into(pt, buf)
        assert res == len(pt)
        assert bytes(buf)[:res] == ct
        encryptor.finalize()
        c = ciphers.Cipher(AES(key), modes.GCM(iv, encryptor.tag), backend)
        decryptor = c.decryptor()
        res = decryptor.update_into(ct, buf)
        decryptor.finalize()
        assert res == len(pt)
        assert bytes(buf)[:res] == pt

    @pytest.mark.supported(only_if=lambda backend: backend.cipher_supported(AES(b'\x00' * 16), modes.GCM(b'0' * 12)), skip_message='Does not support AES GCM')
    def test_finalize_with_tag_already_finalized(self, backend):
        if False:
            for i in range(10):
                print('nop')
        key = binascii.unhexlify(b'e98b72a9881a84ca6b76e0f43e68647a')
        iv = binascii.unhexlify(b'8b23299fde174053f3d652ba')
        encryptor = ciphers.Cipher(AES(key), modes.GCM(iv), backend).encryptor()
        ciphertext = encryptor.update(b'abc') + encryptor.finalize()
        decryptor = ciphers.Cipher(AES(key), modes.GCM(iv, tag=encryptor.tag), backend).decryptor()
        decryptor.update(ciphertext)
        decryptor.finalize()
        with pytest.raises(AlreadyFinalized):
            decryptor.finalize_with_tag(encryptor.tag)

    @pytest.mark.supported(only_if=lambda backend: backend.cipher_supported(AES(b'\x00' * 16), modes.GCM(b'0' * 12)), skip_message='Does not support AES GCM')
    def test_finalize_with_tag_duplicate_tag(self, backend):
        if False:
            print('Hello World!')
        decryptor = ciphers.Cipher(AES(b'\x00' * 16), modes.GCM(b'\x00' * 12, tag=b'\x00' * 16), backend).decryptor()
        with pytest.raises(ValueError):
            decryptor.finalize_with_tag(b'\x00' * 16)

    @pytest.mark.parametrize('params', load_vectors_from_file(os.path.join('ciphers', 'AES', 'ECB', 'ECBGFSbox128.rsp'), load_nist_vectors))
    def test_update_into_multiple_calls(self, params, backend):
        if False:
            while True:
                i = 10
        key = binascii.unhexlify(params['key'])
        pt = binascii.unhexlify(params['plaintext'])
        ct = binascii.unhexlify(params['ciphertext'])
        c = ciphers.Cipher(AES(key), modes.ECB(), backend)
        encryptor = c.encryptor()
        buf = bytearray(len(pt) + 15)
        res = encryptor.update_into(pt[:3], buf)
        assert res == 0
        res = encryptor.update_into(pt[3:], buf)
        assert res == len(pt)
        assert bytes(buf)[:res] == ct

    def test_update_into_buffer_too_small(self, backend):
        if False:
            while True:
                i = 10
        key = b'\x00' * 16
        c = ciphers.Cipher(AES(key), modes.ECB(), backend)
        encryptor = c.encryptor()
        buf = bytearray(16)
        with pytest.raises(ValueError):
            encryptor.update_into(b'testing', buf)

    def test_update_into_immutable(self, backend):
        if False:
            return 10
        key = b'\x00' * 16
        c = ciphers.Cipher(AES(key), modes.ECB(), backend)
        encryptor = c.encryptor()
        buf = b'\x00' * 32
        with pytest.raises((TypeError, BufferError)):
            encryptor.update_into(b'testing', buf)

    @pytest.mark.supported(only_if=lambda backend: backend.cipher_supported(AES(b'\x00' * 16), modes.GCM(b'\x00' * 12)), skip_message='Does not support AES GCM')
    def test_update_into_buffer_too_small_gcm(self, backend):
        if False:
            print('Hello World!')
        key = b'\x00' * 16
        c = ciphers.Cipher(AES(key), modes.GCM(b'\x00' * 12), backend)
        encryptor = c.encryptor()
        buf = bytearray(5)
        with pytest.raises(ValueError):
            encryptor.update_into(b'testing', buf)

@pytest.mark.skipif(sys.platform not in {'linux', 'darwin'}, reason='mmap required')
def test_update_auto_chunking():
    if False:
        while True:
            i = 10
    large_data = mmap.mmap(-1, 2 ** 29 + 2 ** 20, prot=mmap.PROT_READ)
    key = b'\x00' * 16
    c = ciphers.Cipher(AES(key), modes.ECB())
    encryptor = c.encryptor()
    result = encryptor.update(memoryview(large_data))
    assert len(result) == len(large_data)
    decryptor = c.decryptor()
    result = decryptor.update(result)
    assert result == large_data[:]