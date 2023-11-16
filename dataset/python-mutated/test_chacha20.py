import binascii
import os
import struct
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from ...utils import load_nist_vectors
from .utils import _load_all_params

@pytest.mark.supported(only_if=lambda backend: backend.cipher_supported(algorithms.ChaCha20(b'\x00' * 32, b'0' * 16), None), skip_message='Does not support ChaCha20')
class TestChaCha20:

    @pytest.mark.parametrize('vector', _load_all_params(os.path.join('ciphers', 'ChaCha20'), ['counter-overflow.txt', 'rfc7539.txt'], load_nist_vectors))
    def test_vectors(self, vector, backend):
        if False:
            print('Hello World!')
        key = binascii.unhexlify(vector['key'])
        nonce = binascii.unhexlify(vector['nonce'])
        ibc = struct.pack('<Q', int(vector['initial_block_counter']))
        pt = binascii.unhexlify(vector['plaintext'])
        encryptor = Cipher(algorithms.ChaCha20(key, ibc + nonce), None, backend).encryptor()
        computed_ct = encryptor.update(pt) + encryptor.finalize()
        assert binascii.hexlify(computed_ct) == vector['ciphertext']

    def test_buffer_protocol(self, backend):
        if False:
            return 10
        key = bytearray(os.urandom(32))
        nonce = bytearray(os.urandom(16))
        cipher = Cipher(algorithms.ChaCha20(key, nonce), None, backend)
        enc = cipher.encryptor()
        ct = enc.update(bytearray(b'hello')) + enc.finalize()
        dec = cipher.decryptor()
        pt = dec.update(ct) + dec.finalize()
        assert pt == b'hello'

    def test_key_size(self):
        if False:
            return 10
        chacha = algorithms.ChaCha20(b'0' * 32, b'0' * 16)
        assert chacha.key_size == 256

    def test_invalid_key_size(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            algorithms.ChaCha20(b'wrongsize', b'0' * 16)

    def test_invalid_nonce(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            algorithms.ChaCha20(b'0' * 32, b'0')
        with pytest.raises(TypeError):
            algorithms.ChaCha20(b'0' * 32, object())

    def test_invalid_key_type(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError, match='key must be bytes'):
            algorithms.ChaCha20('0' * 32, b'0' * 16)

    def test_partial_blocks(self, backend):
        if False:
            while True:
                i = 10
        key = bytearray(os.urandom(32))
        nonce = bytearray(os.urandom(16))
        cipher = Cipher(algorithms.ChaCha20(key, nonce), None, backend)
        pt = bytearray(os.urandom(96 * 3))
        enc_full = cipher.encryptor()
        ct_full = enc_full.update(pt)
        enc_partial = cipher.encryptor()
        len_partial = len(pt) // 3
        ct_partial_1 = enc_partial.update(pt[:len_partial])
        ct_partial_2 = enc_partial.update(pt[len_partial:len_partial * 2])
        ct_partial_3 = enc_partial.update(pt[len_partial * 2:])
        assert ct_full == ct_partial_1 + ct_partial_2 + ct_partial_3