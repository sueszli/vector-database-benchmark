import binascii
import math
import struct
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from tests.utils import load_nist_vectors
BLOCK_SIZE = 64
MAX_COUNTER = 2 ** 64 - 1

def encrypt(key: bytes, nonce: bytes, initial_block_counter: int, plaintext: bytes) -> bytes:
    if False:
        return 10
    full_nonce = struct.pack('<Q', initial_block_counter) + nonce
    encryptor = Cipher(algorithms.ChaCha20(key, full_nonce), mode=None).encryptor()
    plaintext_len_blocks = math.ceil(len(plaintext) / BLOCK_SIZE)
    blocks_until_overflow = MAX_COUNTER - initial_block_counter + 1
    if plaintext_len_blocks <= blocks_until_overflow:
        return binascii.hexlify(encryptor.update(plaintext))
    else:
        bytes_until_overflow = min(blocks_until_overflow * 64, len(plaintext))
        first_batch = binascii.hexlify(encryptor.update(plaintext[:bytes_until_overflow]))
        full_nonce = struct.pack('<Q', 0) + nonce
        encryptor = Cipher(algorithms.ChaCha20(key, full_nonce), mode=None).encryptor()
        second_batch = binascii.hexlify(encryptor.update(plaintext[bytes_until_overflow:]))
        return first_batch + second_batch

def verify_vectors(filename: Path):
    if False:
        i = 10
        return i + 15
    with open(filename) as f:
        vector_file = f.read().splitlines()
    vectors = load_nist_vectors(vector_file)
    for vector in vectors:
        key = binascii.unhexlify(vector['key'])
        nonce = binascii.unhexlify(vector['nonce'])
        ibc = int(vector['initial_block_counter'])
        pt = binascii.unhexlify(vector['plaintext'])
        computed_ct = encrypt(key, nonce, ibc, pt)
        assert computed_ct == vector['ciphertext']
overflow_path = Path('vectors/cryptography_vectors/ciphers/ChaCha20/counter-overflow.txt')
verify_vectors(overflow_path)