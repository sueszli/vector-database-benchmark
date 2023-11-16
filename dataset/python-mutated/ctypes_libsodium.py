from __future__ import absolute_import, division, print_function, with_statement
import logging
import os
from ctypes import CDLL, c_char_p, c_int, c_ulonglong, byref, create_string_buffer, c_void_p
__all__ = ['ciphers']
libsodium = None
loaded = False
buf_size = 2048
BLOCK_SIZE = 64
lib_path = os.path.dirname(os.path.realpath(__file__))

def load_libsodium():
    if False:
        print('Hello World!')
    global loaded, libsodium, buf
    for p in ('sodium',):
        libsodium_path = os.path.join(lib_path, 'lib', 'libsodium.so')
        if libsodium_path:
            break
    if not libsodium_path:
        raise Exception('libsodium not found')
    logging.info('loading libsodium from %s', libsodium_path)
    libsodium = CDLL(libsodium_path)
    libsodium.sodium_init.restype = c_int
    libsodium.crypto_stream_salsa20_xor_ic.restype = c_int
    libsodium.crypto_stream_salsa20_xor_ic.argtypes = (c_void_p, c_char_p, c_ulonglong, c_char_p, c_ulonglong, c_char_p)
    libsodium.crypto_stream_chacha20_xor_ic.restype = c_int
    libsodium.crypto_stream_chacha20_xor_ic.argtypes = (c_void_p, c_char_p, c_ulonglong, c_char_p, c_ulonglong, c_char_p)
    libsodium.sodium_init()
    buf = create_string_buffer(buf_size)
    loaded = True

class Salsa20Crypto(object):

    def __init__(self, cipher_name, key, iv, op):
        if False:
            return 10
        if not loaded:
            load_libsodium()
        self.key = key
        self.iv = iv
        self.key_ptr = c_char_p(key)
        self.iv_ptr = c_char_p(iv)
        if cipher_name == b'salsa20':
            self.cipher = libsodium.crypto_stream_salsa20_xor_ic
        elif cipher_name == b'chacha20':
            self.cipher = libsodium.crypto_stream_chacha20_xor_ic
        else:
            raise Exception('Unknown cipher')
        self.counter = 0

    def update(self, data):
        if False:
            i = 10
            return i + 15
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
ciphers = {b'salsa20': (32, 8, Salsa20Crypto), b'chacha20': (32, 8, Salsa20Crypto)}

def test_salsa20():
    if False:
        i = 10
        return i + 15
    from ssshare.shadowsocks.crypto import util
    cipher = Salsa20Crypto(b'salsa20', b'k' * 32, b'i' * 16, 1)
    decipher = Salsa20Crypto(b'salsa20', b'k' * 32, b'i' * 16, 0)
    util.run_cipher(cipher, decipher)

def test_chacha20():
    if False:
        return 10
    from ssshare.shadowsocks.crypto import util
    cipher = Salsa20Crypto(b'chacha20', b'k' * 32, b'i' * 16, 1)
    decipher = Salsa20Crypto(b'chacha20', b'k' * 32, b'i' * 16, 0)
    util.run_cipher(cipher, decipher)
if __name__ == '__main__':
    test_chacha20()
    test_salsa20()