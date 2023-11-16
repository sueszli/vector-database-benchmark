from unittest import TestCase
import pytest
import string
import random
from bigdl.dllib.utils.encryption_utils import *

class TestEncryption(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        letters = string.ascii_lowercase
        self.random_str = ''.join((random.choice(letters) for _ in range(100)))

    def test_aes128_cbc_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        enc_bytes = encrypt_bytes_with_AES_CBC(self.random_str.encode(), 'analytics-zoo', 'intel-analytics')
        dec_bytes = decrypt_bytes_with_AES_CBC(enc_bytes, 'analytics-zoo', 'intel-analytics')
        assert dec_bytes == self.random_str.encode(), 'Check AES CBC 128 encryption and decryption result'

    def test_aes256_cbc_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        enc_bytes = encrypt_bytes_with_AES_CBC(self.random_str.encode('utf-8'), 'analytics-zoo', 'intel-analytics', 256)
        dec_bytes = decrypt_bytes_with_AES_CBC(enc_bytes, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_bytes == self.random_str.encode('utf-8'), 'Check AES CBC 256 encryption and decryption result'

    def test_aes128_gcm_bytes(self):
        if False:
            print('Hello World!')
        enc_bytes = encrypt_bytes_with_AES_GCM(self.random_str.encode(), 'analytics-zoo', 'intel-analytics')
        dec_bytes = decrypt_bytes_with_AES_GCM(enc_bytes, 'analytics-zoo', 'intel-analytics')
        assert dec_bytes == self.random_str.encode(), 'Check AES GCM 128 encryption and decryption result'

    def test_aes256_gcm_bytes(self):
        if False:
            while True:
                i = 10
        enc_bytes = encrypt_bytes_with_AES_GCM(self.random_str.encode(), 'analytics-zoo', 'intel-analytics', 256)
        dec_bytes = decrypt_bytes_with_AES_GCM(enc_bytes, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_bytes == self.random_str.encode(), 'Check AES GCM 128 encryption and decryption result'

    def test_aes128_cbc(self):
        if False:
            i = 10
            return i + 15
        enc_str = encrypt_with_AES_CBC(self.random_str, 'analytics-zoo', 'intel-analytics')
        dec_str = decrypt_with_AES_CBC(enc_str, 'analytics-zoo', 'intel-analytics')
        assert dec_str == self.random_str, 'Check AES CBC 128 encryption and decryption result'

    def test_aes256_cbc(self):
        if False:
            while True:
                i = 10
        enc_str = encrypt_with_AES_CBC(self.random_str, 'analytics-zoo', 'intel-analytics', 256)
        dec_str = decrypt_with_AES_CBC(enc_str, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_str == self.random_str, 'Check AES CBC 128 encryption and decryption result'

    def test_aes128_gcm(self):
        if False:
            return 10
        enc_str = encrypt_with_AES_GCM(self.random_str, 'analytics-zoo', 'intel-analytics')
        dec_str = decrypt_with_AES_GCM(enc_str, 'analytics-zoo', 'intel-analytics')
        assert dec_str == self.random_str, 'Check AES GCM 128 encryption and decryption result'

    def test_aes256_gcm(self):
        if False:
            i = 10
            return i + 15
        enc_str = encrypt_with_AES_GCM(self.random_str, 'analytics-zoo', 'intel-analytics', 256)
        dec_str = decrypt_with_AES_GCM(enc_str, 'analytics-zoo', 'intel-analytics', 256)
        assert dec_str == self.random_str, 'Check AES GCM 128 encryption and decryption result'
if __name__ == '__main__':
    pytest.main([__file__])