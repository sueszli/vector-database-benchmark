"""
Stream ciphers.
"""
from scapy.config import conf
from scapy.layers.tls.crypto.common import CipherError
if conf.crypto_valid:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    from cryptography.hazmat.backends import default_backend
_tls_stream_cipher_algs = {}

class _StreamCipherMetaclass(type):
    """
    Cipher classes are automatically registered through this metaclass.
    Furthermore, their name attribute is extracted from their class name.
    """

    def __new__(cls, ciph_name, bases, dct):
        if False:
            return 10
        if ciph_name != '_StreamCipher':
            dct['name'] = ciph_name[7:]
        the_class = super(_StreamCipherMetaclass, cls).__new__(cls, ciph_name, bases, dct)
        if ciph_name != '_StreamCipher':
            _tls_stream_cipher_algs[ciph_name[7:]] = the_class
        return the_class

class _StreamCipher(metaclass=_StreamCipherMetaclass):
    type = 'stream'

    def __init__(self, key=None):
        if False:
            print('Hello World!')
        '\n        Note that we have to keep the encryption/decryption state in unique\n        encryptor and decryptor objects. This differs from _BlockCipher.\n\n        In order to do connection state snapshots, we need to be able to\n        recreate past cipher contexts. This is why we feed _enc_updated_with\n        and _dec_updated_with every time encrypt() or decrypt() is called.\n        '
        self.ready = {'key': True}
        if key is None:
            self.ready['key'] = False
            if hasattr(self, 'expanded_key_len'):
                tmp_len = self.expanded_key_len
            else:
                tmp_len = self.key_len
            key = b'\x00' * tmp_len
        super(_StreamCipher, self).__setattr__('key', key)
        self._cipher = Cipher(self.pc_cls(key), mode=None, backend=default_backend())
        self.encryptor = self._cipher.encryptor()
        self.decryptor = self._cipher.decryptor()
        self._enc_updated_with = b''
        self._dec_updated_with = b''

    def __setattr__(self, name, val):
        if False:
            print('Hello World!')
        '\n        We have to keep the encryptor/decryptor for a long time,\n        however they have to be updated every time the key is changed.\n        '
        if name == 'key':
            if self._cipher is not None:
                self._cipher.algorithm.key = val
                self.encryptor = self._cipher.encryptor()
                self.decryptor = self._cipher.decryptor()
            self.ready['key'] = True
        super(_StreamCipher, self).__setattr__(name, val)

    def encrypt(self, data):
        if False:
            i = 10
            return i + 15
        if False in self.ready.values():
            raise CipherError(data)
        self._enc_updated_with += data
        return self.encryptor.update(data)

    def decrypt(self, data):
        if False:
            for i in range(10):
                print('nop')
        if False in self.ready.values():
            raise CipherError(data)
        self._dec_updated_with += data
        return self.decryptor.update(data)

    def snapshot(self):
        if False:
            print('Hello World!')
        c = self.__class__(self.key)
        c.ready = self.ready.copy()
        c.encryptor.update(self._enc_updated_with)
        c.decryptor.update(self._dec_updated_with)
        c._enc_updated_with = self._enc_updated_with
        c._dec_updated_with = self._dec_updated_with
        return c
if conf.crypto_valid:

    class Cipher_RC4_128(_StreamCipher):
        pc_cls = algorithms.ARC4
        key_len = 16

    class Cipher_RC4_40(Cipher_RC4_128):
        expanded_key_len = 16
        key_len = 5

class Cipher_NULL(_StreamCipher):
    key_len = 0

    def __init__(self, key=None):
        if False:
            for i in range(10):
                print('nop')
        self.ready = {'key': True}
        self._cipher = None
        super(Cipher_NULL, self).__setattr__('key', key)

    def snapshot(self):
        if False:
            i = 10
            return i + 15
        c = self.__class__(self.key)
        c.ready = self.ready.copy()
        return c

    def encrypt(self, data):
        if False:
            print('Hello World!')
        return data

    def decrypt(self, data):
        if False:
            for i in range(10):
                print('nop')
        return data