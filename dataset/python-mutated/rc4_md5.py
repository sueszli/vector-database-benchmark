from __future__ import absolute_import, division, print_function, with_statement
import hashlib
from ssshare.shadowsocks.crypto import openssl
__all__ = ['ciphers']

def create_cipher(alg, key, iv, op, key_as_bytes=0, d=None, salt=None, i=1, padding=1):
    if False:
        print('Hello World!')
    md5 = hashlib.md5()
    md5.update(key)
    md5.update(iv)
    rc4_key = md5.digest()
    return openssl.OpenSSLCrypto(b'rc4', rc4_key, b'', op)
ciphers = {'rc4-md5': (16, 16, create_cipher), 'rc4-md5-6': (16, 6, create_cipher)}

def test():
    if False:
        while True:
            i = 10
    from ssshare.shadowsocks.crypto import util
    cipher = create_cipher('rc4-md5', b'k' * 32, b'i' * 16, 1)
    decipher = create_cipher('rc4-md5', b'k' * 32, b'i' * 16, 0)
    util.run_cipher(cipher, decipher)
if __name__ == '__main__':
    test()