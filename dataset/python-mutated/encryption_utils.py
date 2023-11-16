import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import base64
import hashlib
back_end = default_backend()

def get_private_key(secret_key, salt, key_len=128):
    if False:
        return 10
    '\n    Generate AES required random secret/privacy key\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :return: random key\n    '
    bit_len = key_len / 128 * 16
    return hashlib.pbkdf2_hmac('SHA256', secret_key.encode(), salt.encode(), 65536, int(bit_len))

def encrypt_with_AES_CBC(plain_text, secret_key, salt, key_len=128, block_size=16):
    if False:
        print('Hello World!')
    '\n    encrypt string plain text with AES CBC\n    :param plain_text: plain test in string\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :param block_size: lv size (default 16 for CBC)\n    :return: cipher text in string\n    '
    ct_bytes = encrypt_bytes_with_AES_CBC(plain_text.encode(), secret_key, salt, key_len, block_size)
    return base64.b64encode(ct_bytes).decode()

def decrypt_with_AES_CBC(cipher_text, secret_key, salt, key_len=128, block_size=16):
    if False:
        while True:
            i = 10
    '\n    decrypt string cipher text with AES CBC\n    :param cipher_text: cipher text in string\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :param block_size: lv size (default 16 for CBC)\n    :return: plain test in string\n    '
    plain_bytes = decrypt_bytes_with_AES_CBC(base64.b64decode(cipher_text), secret_key, salt, key_len, block_size)
    return plain_bytes.decode()

def encrypt_bytes_with_AES_CBC(plain_text_bytes, secret_key, salt, key_len=128, block_size=16):
    if False:
        while True:
            i = 10
    '\n    encrypt bytes plain text with AES CBC\n    :param plain_text_bytes: plain test in bytes\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :param block_size: lv size (default 16 for CBC)\n    :return: cipher text in bytes\n    '
    key = get_private_key(secret_key, salt, key_len)
    iv = os.urandom(block_size)
    padder = padding.PKCS7(key_len).padder()
    data = padder.update(plain_text_bytes)
    data += padder.finalize()
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv), backend=back_end).encryptor()
    ct = encryptor.update(data) + encryptor.finalize()
    return iv + ct

def decrypt_bytes_with_AES_CBC(cipher_text_bytes, secret_key, salt, key_len=128, block_size=16):
    if False:
        return 10
    '\n    decrypt bytes cipher text with AES CBC\n    :param cipher_text_bytes: cipher text in bytes\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :param block_size: lv size (default 16 for CBC)\n    :return: plain test in bytes\n    '
    key = get_private_key(secret_key, salt, key_len)
    iv = cipher_text_bytes[:block_size]
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv), backend=back_end).decryptor()
    ct = decryptor.update(cipher_text_bytes[block_size:]) + decryptor.finalize()
    unpadder = padding.PKCS7(key_len).unpadder()
    ct = unpadder.update(ct)
    ct += unpadder.finalize()
    return ct

def encrypt_with_AES_GCM(plain_text, secret_key, salt, key_len=128, block_size=12):
    if False:
        return 10
    '\n    encrypt string plain text with AES GCM\n    :param plain_text: plain test in string\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :param block_size: lv size (default 12 for GCM)\n    :return: cipher text in string\n    '
    ct_bytes = encrypt_bytes_with_AES_GCM(plain_text.encode(), secret_key, salt, key_len, block_size)
    return base64.b64encode(ct_bytes).decode()

def decrypt_with_AES_GCM(cipher_text, secret_key, salt, key_len=128, block_size=12):
    if False:
        while True:
            i = 10
    '\n    decrypt string cipher text with AES GCM\n    :param cipher_text: cipher text in string\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :param block_size: lv size (default 12 for GCM)\n    :return: plain test in string\n    '
    plain_bytes = decrypt_bytes_with_AES_GCM(base64.b64decode(cipher_text), secret_key, salt, key_len, block_size)
    return plain_bytes.decode()

def encrypt_bytes_with_AES_GCM(plain_text_bytes, secret_key, salt, key_len=128, block_size=12):
    if False:
        for i in range(10):
            print('nop')
    '\n    encrypt bytes plain text with AES GCM\n    :param plain_text_bytes: plain test in bytes\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :param block_size: lv size (default 12 for GCM)\n    :return: cipher text in bytes\n    '
    key = get_private_key(secret_key, salt, key_len)
    iv = os.urandom(block_size)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv), backend=back_end).encryptor()
    ct = encryptor.update(plain_text_bytes) + encryptor.finalize()
    return iv + ct + encryptor.tag

def decrypt_bytes_with_AES_GCM(cipher_text_bytes, secret_key, salt, key_len=128, block_size=12):
    if False:
        i = 10
        return i + 15
    '\n    decrypt bytes cipher text with AES GCM\n    :param cipher_text_bytes: cipher text in bytes\n    :param secret_key: secret key in string\n    :param salt: secret slat in string\n    :param key_len: key len (128 or 256)\n    :param block_size: lv size (default 12 for GCM)\n    :return: plain test in bytes\n    '
    key = get_private_key(secret_key, salt, key_len)
    tag = cipher_text_bytes[-16:]
    iv = cipher_text_bytes[:block_size]
    decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=back_end).decryptor()
    ct = decryptor.update(cipher_text_bytes[block_size:-16]) + decryptor.finalize()
    return ct