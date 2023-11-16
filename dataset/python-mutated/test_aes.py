import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base64
from yt_dlp.aes import aes_cbc_decrypt, aes_cbc_decrypt_bytes, aes_cbc_encrypt, aes_ctr_decrypt, aes_ctr_encrypt, aes_decrypt, aes_decrypt_text, aes_ecb_decrypt, aes_ecb_encrypt, aes_encrypt, aes_gcm_decrypt_and_verify, aes_gcm_decrypt_and_verify_bytes, key_expansion, pad_block
from yt_dlp.dependencies import Cryptodome
from yt_dlp.utils import bytes_to_intlist, intlist_to_bytes

class TestAES(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.key = self.iv = [32, 21] + 14 * [0]
        self.secret_msg = b'Secret message goes here'

    def test_encrypt(self):
        if False:
            i = 10
            return i + 15
        msg = b'message'
        key = list(range(16))
        encrypted = aes_encrypt(bytes_to_intlist(msg), key)
        decrypted = intlist_to_bytes(aes_decrypt(encrypted, key))
        self.assertEqual(decrypted, msg)

    def test_cbc_decrypt(self):
        if False:
            print('Hello World!')
        data = b"\x97\x92+\xe5\x0b\xc3\x18\x91ky9m&\xb3\xb5@\xe6'\xc2\x96.\xc8u\x88\xab9-[\x9e|\xf1\xcd"
        decrypted = intlist_to_bytes(aes_cbc_decrypt(bytes_to_intlist(data), self.key, self.iv))
        self.assertEqual(decrypted.rstrip(b'\x08'), self.secret_msg)
        if Cryptodome.AES:
            decrypted = aes_cbc_decrypt_bytes(data, intlist_to_bytes(self.key), intlist_to_bytes(self.iv))
            self.assertEqual(decrypted.rstrip(b'\x08'), self.secret_msg)

    def test_cbc_encrypt(self):
        if False:
            for i in range(10):
                print('nop')
        data = bytes_to_intlist(self.secret_msg)
        encrypted = intlist_to_bytes(aes_cbc_encrypt(data, self.key, self.iv))
        self.assertEqual(encrypted, b"\x97\x92+\xe5\x0b\xc3\x18\x91ky9m&\xb3\xb5@\xe6'\xc2\x96.\xc8u\x88\xab9-[\x9e|\xf1\xcd")

    def test_ctr_decrypt(self):
        if False:
            while True:
                i = 10
        data = bytes_to_intlist(b'\x03\xc7\xdd\xd4\x8e\xb3\xbc\x1a*O\xdc1\x12+8Aio\xd1z\xb5#\xaf\x08')
        decrypted = intlist_to_bytes(aes_ctr_decrypt(data, self.key, self.iv))
        self.assertEqual(decrypted.rstrip(b'\x08'), self.secret_msg)

    def test_ctr_encrypt(self):
        if False:
            return 10
        data = bytes_to_intlist(self.secret_msg)
        encrypted = intlist_to_bytes(aes_ctr_encrypt(data, self.key, self.iv))
        self.assertEqual(encrypted, b'\x03\xc7\xdd\xd4\x8e\xb3\xbc\x1a*O\xdc1\x12+8Aio\xd1z\xb5#\xaf\x08')

    def test_gcm_decrypt(self):
        if False:
            for i in range(10):
                print('nop')
        data = b'\x159Y\xcf5eud\x90\x9c\x85&]\x14\x1d\x0f.\x08\xb4T\xe4/\x17\xbd'
        authentication_tag = b'\xe8&I\x80rI\x07\x9d}YWuU@:e'
        decrypted = intlist_to_bytes(aes_gcm_decrypt_and_verify(bytes_to_intlist(data), self.key, bytes_to_intlist(authentication_tag), self.iv[:12]))
        self.assertEqual(decrypted.rstrip(b'\x08'), self.secret_msg)
        if Cryptodome.AES:
            decrypted = aes_gcm_decrypt_and_verify_bytes(data, intlist_to_bytes(self.key), authentication_tag, intlist_to_bytes(self.iv[:12]))
            self.assertEqual(decrypted.rstrip(b'\x08'), self.secret_msg)

    def test_decrypt_text(self):
        if False:
            print('Hello World!')
        password = intlist_to_bytes(self.key).decode()
        encrypted = base64.b64encode(intlist_to_bytes(self.iv[:8]) + b'\x17\x15\x93\xab\x8d\x80V\xcdV\xe0\t\xcdo\xc2\xa5\xd8ksM\r\xe27N\xae').decode()
        decrypted = aes_decrypt_text(encrypted, password, 16)
        self.assertEqual(decrypted, self.secret_msg)
        password = intlist_to_bytes(self.key).decode()
        encrypted = base64.b64encode(intlist_to_bytes(self.iv[:8]) + b'\x0b\xe6\xa4\xd9z\x0e\xb8\xb9\xd0\xd4i_\x85\x1d\x99\x98_\xe5\x80\xe7.\xbf\xa5\x83').decode()
        decrypted = aes_decrypt_text(encrypted, password, 32)
        self.assertEqual(decrypted, self.secret_msg)

    def test_ecb_encrypt(self):
        if False:
            for i in range(10):
                print('nop')
        data = bytes_to_intlist(self.secret_msg)
        encrypted = intlist_to_bytes(aes_ecb_encrypt(data, self.key))
        self.assertEqual(encrypted, b'\xaa\x86]\x81\x97>\x02\x92\x9d\x1bR[[L/u\xd3&\xd1(h\xde{\x81\x94\xba\x02\xae\xbd\xa6\xd0:')

    def test_ecb_decrypt(self):
        if False:
            while True:
                i = 10
        data = bytes_to_intlist(b'\xaa\x86]\x81\x97>\x02\x92\x9d\x1bR[[L/u\xd3&\xd1(h\xde{\x81\x94\xba\x02\xae\xbd\xa6\xd0:')
        decrypted = intlist_to_bytes(aes_ecb_decrypt(data, self.key, self.iv))
        self.assertEqual(decrypted.rstrip(b'\x08'), self.secret_msg)

    def test_key_expansion(self):
        if False:
            return 10
        key = '4f6bdaa39e2f8cb07f5e722d9edef314'
        self.assertEqual(key_expansion(bytes_to_intlist(bytearray.fromhex(key))), [79, 107, 218, 163, 158, 47, 140, 176, 127, 94, 114, 45, 158, 222, 243, 20, 83, 102, 32, 168, 205, 73, 172, 24, 178, 23, 222, 53, 44, 201, 45, 33, 140, 190, 221, 217, 65, 247, 113, 193, 243, 224, 175, 244, 223, 41, 130, 213, 45, 173, 222, 71, 108, 90, 175, 134, 159, 186, 0, 114, 64, 147, 130, 167, 249, 190, 130, 78, 149, 228, 45, 200, 10, 94, 45, 186, 74, 205, 175, 29, 84, 199, 38, 152, 193, 35, 11, 80, 203, 125, 38, 234, 129, 176, 137, 247, 147, 96, 78, 148, 82, 67, 69, 196, 153, 62, 99, 46, 24, 142, 234, 217, 202, 231, 123, 57, 152, 164, 62, 253, 1, 154, 93, 211, 25, 20, 183, 10, 176, 78, 28, 237, 40, 234, 34, 16, 41, 112, 127, 195, 48, 100, 200, 201, 232, 166, 193, 233, 192, 76, 227, 249, 233, 60, 156, 58, 217, 88, 84, 243, 180, 134, 204, 220, 116, 202, 47, 37, 157, 246, 179, 31, 68, 174, 231, 236])

    def test_pad_block(self):
        if False:
            while True:
                i = 10
        block = [33, 160, 67, 255]
        self.assertEqual(pad_block(block, 'pkcs7'), block + [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12])
        self.assertEqual(pad_block(block, 'iso7816'), block + [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(pad_block(block, 'whitespace'), block + [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32])
        self.assertEqual(pad_block(block, 'zero'), block + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        block = list(range(16))
        for mode in ('pkcs7', 'iso7816', 'whitespace', 'zero'):
            self.assertEqual(pad_block(block, mode), block, mode)
if __name__ == '__main__':
    unittest.main()