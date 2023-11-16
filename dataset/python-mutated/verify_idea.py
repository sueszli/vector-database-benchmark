import binascii
import botan
from tests.utils import load_nist_vectors
BLOCK_SIZE = 64

def encrypt(mode, key, iv, plaintext):
    if False:
        while True:
            i = 10
    encryptor = botan.Cipher(f'IDEA/{mode}/NoPadding', 'encrypt', binascii.unhexlify(key))
    cipher_text = encryptor.cipher(binascii.unhexlify(plaintext), binascii.unhexlify(iv))
    return binascii.hexlify(cipher_text)

def verify_vectors(mode, filename):
    if False:
        i = 10
        return i + 15
    with open(filename) as f:
        vector_file = f.read().splitlines()
    vectors = load_nist_vectors(vector_file)
    for vector in vectors:
        ct = encrypt(mode, vector['key'], vector['iv'], vector['plaintext'])
        assert ct == vector['ciphertext']
cbc_path = 'tests/hazmat/primitives/vectors/ciphers/IDEA/idea-cbc.txt'
verify_vectors('CBC', cbc_path)
ofb_path = 'tests/hazmat/primitives/vectors/ciphers/IDEA/idea-ofb.txt'
verify_vectors('OFB', ofb_path)
cfb_path = 'tests/hazmat/primitives/vectors/ciphers/IDEA/idea-cfb.txt'
verify_vectors('CFB', cfb_path)