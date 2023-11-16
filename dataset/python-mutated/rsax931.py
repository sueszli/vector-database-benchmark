"""
Create and verify ANSI X9.31 RSA signatures using OpenSSL libcrypto
"""
import ctypes.util
import glob
import os
import platform
import sys
from ctypes import c_char_p, c_int, c_void_p, cdll, create_string_buffer, pointer
import salt.utils.platform
import salt.utils.stringutils
OPENSSL_INIT_ADD_ALL_CIPHERS = 4
OPENSSL_INIT_ADD_ALL_DIGESTS = 8
OPENSSL_INIT_NO_LOAD_CONFIG = 128

def _find_libcrypto():
    if False:
        return 10
    '\n    Find the path (or return the short name) of libcrypto.\n    '
    if sys.platform.startswith('win'):
        lib = None
        for path in sys.path:
            lib = glob.glob(os.path.join(path, 'libcrypto*.dll'))
            lib = lib[0] if lib else None
            if lib:
                break
    elif salt.utils.platform.is_darwin():
        lib = glob.glob('/opt/salt/lib/libcrypto.dylib')
        lib = lib or glob.glob('lib/libcrypto.dylib')
        brew_prefix = os.getenv('HOMEBREW_PREFIX', '/usr/local')
        lib = lib or glob.glob(os.path.join(brew_prefix, 'opt/openssl/lib/libcrypto.dylib'))
        lib = lib or glob.glob(os.path.join(brew_prefix, 'opt/openssl@*/lib/libcrypto.dylib'))
        lib = lib or glob.glob('/opt/local/lib/libcrypto.dylib')
        if platform.mac_ver()[0].split('.')[:2] == ['10', '15']:
            lib = lib or glob.glob('/usr/lib/libcrypto.*.dylib')
            lib = list(reversed(sorted(lib)))
        elif int(platform.mac_ver()[0].split('.')[0]) < 11:
            lib = lib or ['/usr/lib/libcrypto.dylib']
        lib = lib[0] if lib else None
    elif getattr(sys, 'frozen', False) and salt.utils.platform.is_smartos():
        lib = glob.glob(os.path.join(os.path.dirname(sys.executable), 'libcrypto.so*'))
        lib = lib[0] if lib else None
    else:
        lib = ctypes.util.find_library('crypto')
        if not lib:
            if salt.utils.platform.is_sunos():
                lib = glob.glob('/opt/saltstack/salt/run/libcrypto.so*')
                lib = lib or glob.glob('/opt/local/lib/libcrypto.so*')
                lib = lib or glob.glob('/opt/tools/lib/libcrypto.so*')
                lib = lib[0] if lib else None
            elif salt.utils.platform.is_aix():
                if os.path.isdir('/opt/saltstack/salt/run') or os.path.isdir('/opt/salt/lib'):
                    lib = glob.glob('/opt/saltstack/salt/run/libcrypto.so*')
                    lib = lib or glob.glob('/opt/salt/lib/libcrypto.so*')
                else:
                    lib = glob.glob('/opt/freeware/lib/libcrypto.so*')
                lib = lib[0] if lib else None
    if not lib:
        raise OSError('Cannot locate OpenSSL libcrypto')
    return lib

def _load_libcrypto():
    if False:
        return 10
    '\n    Attempt to load libcrypto.\n    '
    return cdll.LoadLibrary(_find_libcrypto())

def _init_libcrypto():
    if False:
        while True:
            i = 10
    '\n    Set up libcrypto argtypes and initialize the library\n    '
    libcrypto = _load_libcrypto()
    try:
        openssl_version_num = libcrypto.OpenSSL_version_num
        if callable(openssl_version_num):
            openssl_version_num = openssl_version_num()
        if openssl_version_num < 269484032:
            libcrypto.OPENSSL_init_crypto()
    except AttributeError:
        libcrypto.OPENSSL_no_config()
        libcrypto.OPENSSL_add_all_algorithms_noconf()
    libcrypto.RSA_new.argtypes = ()
    libcrypto.RSA_new.restype = c_void_p
    libcrypto.RSA_free.argtypes = (c_void_p,)
    libcrypto.RSA_size.argtype = c_void_p
    libcrypto.BIO_new_mem_buf.argtypes = (c_char_p, c_int)
    libcrypto.BIO_new_mem_buf.restype = c_void_p
    libcrypto.BIO_free.argtypes = (c_void_p,)
    libcrypto.PEM_read_bio_RSAPrivateKey.argtypes = (c_void_p, c_void_p, c_void_p, c_void_p)
    libcrypto.PEM_read_bio_RSAPrivateKey.restype = c_void_p
    libcrypto.PEM_read_bio_RSA_PUBKEY.argtypes = (c_void_p, c_void_p, c_void_p, c_void_p)
    libcrypto.PEM_read_bio_RSA_PUBKEY.restype = c_void_p
    libcrypto.RSA_private_encrypt.argtypes = (c_int, c_char_p, c_char_p, c_void_p, c_int)
    libcrypto.RSA_public_decrypt.argtypes = (c_int, c_char_p, c_char_p, c_void_p, c_int)
    return libcrypto
libcrypto = _init_libcrypto()
RSA_X931_PADDING = 5

class RSAX931Signer:
    """
    Create ANSI X9.31 RSA signatures using OpenSSL libcrypto
    """

    def __init__(self, keydata):
        if False:
            i = 10
            return i + 15
        '\n        Init an RSAX931Signer instance\n\n        :param str keydata: The RSA private key in PEM format\n        '
        keydata = salt.utils.stringutils.to_bytes(keydata, 'ascii')
        self._bio = libcrypto.BIO_new_mem_buf(keydata, len(keydata))
        self._rsa = c_void_p(libcrypto.RSA_new())
        if not libcrypto.PEM_read_bio_RSAPrivateKey(self._bio, pointer(self._rsa), None, None):
            raise ValueError('invalid RSA private key')

    def __del__(self):
        if False:
            while True:
                i = 10
        libcrypto.BIO_free(self._bio)
        libcrypto.RSA_free(self._rsa)

    def sign(self, msg):
        if False:
            print('Hello World!')
        '\n        Sign a message (digest) using the private key\n\n        :param str msg: The message (digest) to sign\n        :rtype: str\n        :return: The signature, or an empty string if the encryption failed\n        '
        buf = create_string_buffer(libcrypto.RSA_size(self._rsa))
        msg = salt.utils.stringutils.to_bytes(msg)
        size = libcrypto.RSA_private_encrypt(len(msg), msg, buf, self._rsa, RSA_X931_PADDING)
        if size < 0:
            raise ValueError('Unable to encrypt message')
        return buf[0:size]

class RSAX931Verifier:
    """
    Verify ANSI X9.31 RSA signatures using OpenSSL libcrypto
    """

    def __init__(self, pubdata):
        if False:
            return 10
        '\n        Init an RSAX931Verifier instance\n\n        :param str pubdata: The RSA public key in PEM format\n        '
        pubdata = salt.utils.stringutils.to_bytes(pubdata, 'ascii')
        pubdata = pubdata.replace(b'RSA ', b'')
        self._bio = libcrypto.BIO_new_mem_buf(pubdata, len(pubdata))
        self._rsa = c_void_p(libcrypto.RSA_new())
        if not libcrypto.PEM_read_bio_RSA_PUBKEY(self._bio, pointer(self._rsa), None, None):
            raise ValueError('invalid RSA public key')

    def __del__(self):
        if False:
            return 10
        libcrypto.BIO_free(self._bio)
        libcrypto.RSA_free(self._rsa)

    def verify(self, signed):
        if False:
            print('Hello World!')
        '\n        Recover the message (digest) from the signature using the public key\n\n        :param str signed: The signature created with the private key\n        :rtype: str\n        :return: The message (digest) recovered from the signature, or an empty\n            string if the decryption failed\n        '
        buf = create_string_buffer(libcrypto.RSA_size(self._rsa))
        signed = salt.utils.stringutils.to_bytes(signed)
        size = libcrypto.RSA_public_decrypt(len(signed), signed, buf, self._rsa, RSA_X931_PADDING)
        if size < 0:
            raise ValueError('Unable to decrypt message')
        return buf[0:size]