from __future__ import absolute_import, division, print_function, with_statement
from ctypes import c_char_p, c_int, c_ulong, c_ulonglong, byref, create_string_buffer, c_void_p, CDLL
from ssshare.shadowsocks.crypto import util
import os
lib_path = os.path.dirname(os.path.realpath(__file__))
__all__ = ['ciphers']
libsodium = None
loaded = False
buf_size = 2048
BLOCK_SIZE = 64

def load_libsodium():
    if False:
        while True:
            i = 10
    global loaded, libsodium, buf
    libsodium = CDLL(os.path.join(lib_path, 'lib', 'libsodium.so'))
    if libsodium is None:
        raise Exception('libsodium not found')
    libsodium.crypto_stream_salsa20_xor_ic.restype = c_int
    libsodium.crypto_stream_salsa20_xor_ic.argtypes = (c_void_p, c_char_p, c_ulonglong, c_char_p, c_ulonglong, c_char_p)
    libsodium.crypto_stream_chacha20_xor_ic.restype = c_int
    libsodium.crypto_stream_chacha20_xor_ic.argtypes = (c_void_p, c_char_p, c_ulonglong, c_char_p, c_ulonglong, c_char_p)
    try:
        libsodium.crypto_stream_chacha20_ietf_xor_ic.restype = c_int
        libsodium.crypto_stream_chacha20_ietf_xor_ic.argtypes = (c_void_p, c_char_p, c_ulonglong, c_char_p, c_ulong, c_char_p)
    except:
        pass
    buf = create_string_buffer(buf_size)
    loaded = True

class SodiumCrypto(object):

    def __init__(self, cipher_name, key, iv, op):
        if False:
            return 10
        if not loaded:
            load_libsodium()
        self.key = key
        self.iv = iv
        self.key_ptr = c_char_p(key)
        self.iv_ptr = c_char_p(iv)
        if cipher_name == 'salsa20':
            self.cipher = libsodium.crypto_stream_salsa20_xor_ic
        elif cipher_name == 'chacha20':
            self.cipher = libsodium.crypto_stream_chacha20_xor_ic
        elif cipher_name == 'chacha20-ietf':
            self.cipher = libsodium.crypto_stream_chacha20_ietf_xor_ic
        else:
            raise Exception('Unknown cipher')
        self.counter = 0

    def update(self, data):
        if False:
            print('Hello World!')
        global buf_size, buf
        l = len(data)
        padding = self.counter % BLOCK_SIZE
        if buf_size < padding + l:
            buf_size = (padding + l) * 2
            buf = create_string_buffer(buf_size)
        if padding:
            data = b'\x00' * padding + data
        self.cipher(byref(buf), c_char_p(data), padding + l, self.iv_ptr, int(self.counter / BLOCK_SIZE), self.key_ptr)
        self.counter += l
        return buf.raw[padding:padding + l]
ciphers = {'salsa20': (32, 8, SodiumCrypto), 'chacha20': (32, 8, SodiumCrypto), 'chacha20-ietf': (32, 12, SodiumCrypto)}

def test_salsa20():
    if False:
        i = 10
        return i + 15
    cipher = SodiumCrypto('salsa20', b'k' * 32, b'i' * 16, 1)
    decipher = SodiumCrypto('salsa20', b'k' * 32, b'i' * 16, 0)
    util.run_cipher(cipher, decipher)

def test_chacha20():
    if False:
        return 10
    cipher = SodiumCrypto('chacha20', b'k' * 32, b'i' * 16, 1)
    decipher = SodiumCrypto('chacha20', b'k' * 32, b'i' * 16, 0)
    util.run_cipher(cipher, decipher)

def test_chacha20_ietf():
    if False:
        while True:
            i = 10
    cipher = SodiumCrypto('chacha20-ietf', b'k' * 32, b'i' * 16, 1)
    decipher = SodiumCrypto('chacha20-ietf', b'k' * 32, b'i' * 16, 0)
    util.run_cipher(cipher, decipher)
if __name__ == '__main__':
    test_chacha20_ietf()
    test_chacha20()
    test_salsa20()