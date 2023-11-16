import base64
from math import ceil
from .compat import compat_ord
from .dependencies import Cryptodome
from .utils import bytes_to_intlist, intlist_to_bytes
if Cryptodome.AES:

    def aes_cbc_decrypt_bytes(data, key, iv):
        if False:
            return 10
        ' Decrypt bytes with AES-CBC using pycryptodome '
        return Cryptodome.AES.new(key, Cryptodome.AES.MODE_CBC, iv).decrypt(data)

    def aes_gcm_decrypt_and_verify_bytes(data, key, tag, nonce):
        if False:
            for i in range(10):
                print('nop')
        ' Decrypt bytes with AES-GCM using pycryptodome '
        return Cryptodome.AES.new(key, Cryptodome.AES.MODE_GCM, nonce).decrypt_and_verify(data, tag)
else:

    def aes_cbc_decrypt_bytes(data, key, iv):
        if False:
            i = 10
            return i + 15
        ' Decrypt bytes with AES-CBC using native implementation since pycryptodome is unavailable '
        return intlist_to_bytes(aes_cbc_decrypt(*map(bytes_to_intlist, (data, key, iv))))

    def aes_gcm_decrypt_and_verify_bytes(data, key, tag, nonce):
        if False:
            print('Hello World!')
        ' Decrypt bytes with AES-GCM using native implementation since pycryptodome is unavailable '
        return intlist_to_bytes(aes_gcm_decrypt_and_verify(*map(bytes_to_intlist, (data, key, tag, nonce))))

def aes_cbc_encrypt_bytes(data, key, iv, **kwargs):
    if False:
        print('Hello World!')
    return intlist_to_bytes(aes_cbc_encrypt(*map(bytes_to_intlist, (data, key, iv)), **kwargs))
BLOCK_SIZE_BYTES = 16

def unpad_pkcs7(data):
    if False:
        print('Hello World!')
    return data[:-compat_ord(data[-1])]

def pkcs7_padding(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    PKCS#7 padding\n\n    @param {int[]} data        cleartext\n    @returns {int[]}           padding data\n    '
    remaining_length = BLOCK_SIZE_BYTES - len(data) % BLOCK_SIZE_BYTES
    return data + [remaining_length] * remaining_length

def pad_block(block, padding_mode):
    if False:
        return 10
    '\n    Pad a block with the given padding mode\n    @param {int[]} block        block to pad\n    @param padding_mode         padding mode\n    '
    padding_size = BLOCK_SIZE_BYTES - len(block)
    PADDING_BYTE = {'pkcs7': padding_size, 'iso7816': 0, 'whitespace': 32, 'zero': 0}
    if padding_size < 0:
        raise ValueError('Block size exceeded')
    elif padding_mode not in PADDING_BYTE:
        raise NotImplementedError(f'Padding mode {padding_mode} is not implemented')
    if padding_mode == 'iso7816' and padding_size:
        block = block + [128]
        padding_size -= 1
    return block + [PADDING_BYTE[padding_mode]] * padding_size

def aes_ecb_encrypt(data, key, iv=None):
    if False:
        return 10
    '\n    Encrypt with aes in ECB mode. Using PKCS#7 padding\n\n    @param {int[]} data        cleartext\n    @param {int[]} key         16/24/32-Byte cipher key\n    @param {int[]} iv          Unused for this mode\n    @returns {int[]}           encrypted data\n    '
    expanded_key = key_expansion(key)
    block_count = int(ceil(float(len(data)) / BLOCK_SIZE_BYTES))
    encrypted_data = []
    for i in range(block_count):
        block = data[i * BLOCK_SIZE_BYTES:(i + 1) * BLOCK_SIZE_BYTES]
        encrypted_data += aes_encrypt(pkcs7_padding(block), expanded_key)
    return encrypted_data

def aes_ecb_decrypt(data, key, iv=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decrypt with aes in ECB mode\n\n    @param {int[]} data        cleartext\n    @param {int[]} key         16/24/32-Byte cipher key\n    @param {int[]} iv          Unused for this mode\n    @returns {int[]}           decrypted data\n    '
    expanded_key = key_expansion(key)
    block_count = int(ceil(float(len(data)) / BLOCK_SIZE_BYTES))
    encrypted_data = []
    for i in range(block_count):
        block = data[i * BLOCK_SIZE_BYTES:(i + 1) * BLOCK_SIZE_BYTES]
        encrypted_data += aes_decrypt(block, expanded_key)
    encrypted_data = encrypted_data[:len(data)]
    return encrypted_data

def aes_ctr_decrypt(data, key, iv):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decrypt with aes in counter mode\n\n    @param {int[]} data        cipher\n    @param {int[]} key         16/24/32-Byte cipher key\n    @param {int[]} iv          16-Byte initialization vector\n    @returns {int[]}           decrypted data\n    '
    return aes_ctr_encrypt(data, key, iv)

def aes_ctr_encrypt(data, key, iv):
    if False:
        for i in range(10):
            print('nop')
    '\n    Encrypt with aes in counter mode\n\n    @param {int[]} data        cleartext\n    @param {int[]} key         16/24/32-Byte cipher key\n    @param {int[]} iv          16-Byte initialization vector\n    @returns {int[]}           encrypted data\n    '
    expanded_key = key_expansion(key)
    block_count = int(ceil(float(len(data)) / BLOCK_SIZE_BYTES))
    counter = iter_vector(iv)
    encrypted_data = []
    for i in range(block_count):
        counter_block = next(counter)
        block = data[i * BLOCK_SIZE_BYTES:(i + 1) * BLOCK_SIZE_BYTES]
        block += [0] * (BLOCK_SIZE_BYTES - len(block))
        cipher_counter_block = aes_encrypt(counter_block, expanded_key)
        encrypted_data += xor(block, cipher_counter_block)
    encrypted_data = encrypted_data[:len(data)]
    return encrypted_data

def aes_cbc_decrypt(data, key, iv):
    if False:
        return 10
    '\n    Decrypt with aes in CBC mode\n\n    @param {int[]} data        cipher\n    @param {int[]} key         16/24/32-Byte cipher key\n    @param {int[]} iv          16-Byte IV\n    @returns {int[]}           decrypted data\n    '
    expanded_key = key_expansion(key)
    block_count = int(ceil(float(len(data)) / BLOCK_SIZE_BYTES))
    decrypted_data = []
    previous_cipher_block = iv
    for i in range(block_count):
        block = data[i * BLOCK_SIZE_BYTES:(i + 1) * BLOCK_SIZE_BYTES]
        block += [0] * (BLOCK_SIZE_BYTES - len(block))
        decrypted_block = aes_decrypt(block, expanded_key)
        decrypted_data += xor(decrypted_block, previous_cipher_block)
        previous_cipher_block = block
    decrypted_data = decrypted_data[:len(data)]
    return decrypted_data

def aes_cbc_encrypt(data, key, iv, *, padding_mode='pkcs7'):
    if False:
        return 10
    '\n    Encrypt with aes in CBC mode\n\n    @param {int[]} data        cleartext\n    @param {int[]} key         16/24/32-Byte cipher key\n    @param {int[]} iv          16-Byte IV\n    @param padding_mode        Padding mode to use\n    @returns {int[]}           encrypted data\n    '
    expanded_key = key_expansion(key)
    block_count = int(ceil(float(len(data)) / BLOCK_SIZE_BYTES))
    encrypted_data = []
    previous_cipher_block = iv
    for i in range(block_count):
        block = data[i * BLOCK_SIZE_BYTES:(i + 1) * BLOCK_SIZE_BYTES]
        block = pad_block(block, padding_mode)
        mixed_block = xor(block, previous_cipher_block)
        encrypted_block = aes_encrypt(mixed_block, expanded_key)
        encrypted_data += encrypted_block
        previous_cipher_block = encrypted_block
    return encrypted_data

def aes_gcm_decrypt_and_verify(data, key, tag, nonce):
    if False:
        print('Hello World!')
    '\n    Decrypt with aes in GBM mode and checks authenticity using tag\n\n    @param {int[]} data        cipher\n    @param {int[]} key         16-Byte cipher key\n    @param {int[]} tag         authentication tag\n    @param {int[]} nonce       IV (recommended 12-Byte)\n    @returns {int[]}           decrypted data\n    '
    hash_subkey = aes_encrypt([0] * BLOCK_SIZE_BYTES, key_expansion(key))
    if len(nonce) == 12:
        j0 = nonce + [0, 0, 0, 1]
    else:
        fill = (BLOCK_SIZE_BYTES - len(nonce) % BLOCK_SIZE_BYTES) % BLOCK_SIZE_BYTES + 8
        ghash_in = nonce + [0] * fill + bytes_to_intlist((8 * len(nonce)).to_bytes(8, 'big'))
        j0 = ghash(hash_subkey, ghash_in)
    iv_ctr = inc(j0)
    decrypted_data = aes_ctr_decrypt(data, key, iv_ctr + [0] * (BLOCK_SIZE_BYTES - len(iv_ctr)))
    pad_len = len(data) // 16 * 16
    s_tag = ghash(hash_subkey, data + [0] * (BLOCK_SIZE_BYTES - len(data) + pad_len) + bytes_to_intlist((0 * 8).to_bytes(8, 'big') + (len(data) * 8).to_bytes(8, 'big')))
    if tag != aes_ctr_encrypt(s_tag, key, j0):
        raise ValueError('Mismatching authentication tag')
    return decrypted_data

def aes_encrypt(data, expanded_key):
    if False:
        while True:
            i = 10
    '\n    Encrypt one block with aes\n\n    @param {int[]} data          16-Byte state\n    @param {int[]} expanded_key  176/208/240-Byte expanded key\n    @returns {int[]}             16-Byte cipher\n    '
    rounds = len(expanded_key) // BLOCK_SIZE_BYTES - 1
    data = xor(data, expanded_key[:BLOCK_SIZE_BYTES])
    for i in range(1, rounds + 1):
        data = sub_bytes(data)
        data = shift_rows(data)
        if i != rounds:
            data = list(iter_mix_columns(data, MIX_COLUMN_MATRIX))
        data = xor(data, expanded_key[i * BLOCK_SIZE_BYTES:(i + 1) * BLOCK_SIZE_BYTES])
    return data

def aes_decrypt(data, expanded_key):
    if False:
        while True:
            i = 10
    '\n    Decrypt one block with aes\n\n    @param {int[]} data          16-Byte cipher\n    @param {int[]} expanded_key  176/208/240-Byte expanded key\n    @returns {int[]}             16-Byte state\n    '
    rounds = len(expanded_key) // BLOCK_SIZE_BYTES - 1
    for i in range(rounds, 0, -1):
        data = xor(data, expanded_key[i * BLOCK_SIZE_BYTES:(i + 1) * BLOCK_SIZE_BYTES])
        if i != rounds:
            data = list(iter_mix_columns(data, MIX_COLUMN_MATRIX_INV))
        data = shift_rows_inv(data)
        data = sub_bytes_inv(data)
    data = xor(data, expanded_key[:BLOCK_SIZE_BYTES])
    return data

def aes_decrypt_text(data, password, key_size_bytes):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decrypt text\n    - The first 8 Bytes of decoded 'data' are the 8 high Bytes of the counter\n    - The cipher key is retrieved by encrypting the first 16 Byte of 'password'\n      with the first 'key_size_bytes' Bytes from 'password' (if necessary filled with 0's)\n    - Mode of operation is 'counter'\n\n    @param {str} data                    Base64 encoded string\n    @param {str,unicode} password        Password (will be encoded with utf-8)\n    @param {int} key_size_bytes          Possible values: 16 for 128-Bit, 24 for 192-Bit or 32 for 256-Bit\n    @returns {str}                       Decrypted data\n    "
    NONCE_LENGTH_BYTES = 8
    data = bytes_to_intlist(base64.b64decode(data))
    password = bytes_to_intlist(password.encode())
    key = password[:key_size_bytes] + [0] * (key_size_bytes - len(password))
    key = aes_encrypt(key[:BLOCK_SIZE_BYTES], key_expansion(key)) * (key_size_bytes // BLOCK_SIZE_BYTES)
    nonce = data[:NONCE_LENGTH_BYTES]
    cipher = data[NONCE_LENGTH_BYTES:]
    decrypted_data = aes_ctr_decrypt(cipher, key, nonce + [0] * (BLOCK_SIZE_BYTES - NONCE_LENGTH_BYTES))
    plaintext = intlist_to_bytes(decrypted_data)
    return plaintext
RCON = (141, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54)
SBOX = (99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22)
SBOX_INV = (82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251, 124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78, 8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37, 114, 248, 246, 100, 134, 104, 152, 22, 212, 164, 92, 204, 93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132, 144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189, 3, 1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115, 150, 172, 116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98, 14, 170, 24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244, 31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125)
MIX_COLUMN_MATRIX = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))
MIX_COLUMN_MATRIX_INV = ((14, 11, 13, 9), (9, 14, 11, 13), (13, 9, 14, 11), (11, 13, 9, 14))
RIJNDAEL_EXP_TABLE = (1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53, 95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170, 229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49, 83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205, 76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136, 131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154, 181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163, 254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160, 251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65, 195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117, 159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128, 155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84, 252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202, 69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14, 18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23, 57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1)
RIJNDAEL_LOG_TABLE = (0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3, 100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193, 125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120, 101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142, 150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56, 102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16, 126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186, 43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87, 175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232, 44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160, 127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183, 204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157, 151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209, 83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171, 68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165, 103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7)

def key_expansion(data):
    if False:
        print('Hello World!')
    '\n    Generate key schedule\n\n    @param {int[]} data  16/24/32-Byte cipher key\n    @returns {int[]}     176/208/240-Byte expanded key\n    '
    data = data[:]
    rcon_iteration = 1
    key_size_bytes = len(data)
    expanded_key_size_bytes = (key_size_bytes // 4 + 7) * BLOCK_SIZE_BYTES
    while len(data) < expanded_key_size_bytes:
        temp = data[-4:]
        temp = key_schedule_core(temp, rcon_iteration)
        rcon_iteration += 1
        data += xor(temp, data[-key_size_bytes:4 - key_size_bytes])
        for _ in range(3):
            temp = data[-4:]
            data += xor(temp, data[-key_size_bytes:4 - key_size_bytes])
        if key_size_bytes == 32:
            temp = data[-4:]
            temp = sub_bytes(temp)
            data += xor(temp, data[-key_size_bytes:4 - key_size_bytes])
        for _ in range(3 if key_size_bytes == 32 else 2 if key_size_bytes == 24 else 0):
            temp = data[-4:]
            data += xor(temp, data[-key_size_bytes:4 - key_size_bytes])
    data = data[:expanded_key_size_bytes]
    return data

def iter_vector(iv):
    if False:
        for i in range(10):
            print('nop')
    while True:
        yield iv
        iv = inc(iv)

def sub_bytes(data):
    if False:
        return 10
    return [SBOX[x] for x in data]

def sub_bytes_inv(data):
    if False:
        for i in range(10):
            print('nop')
    return [SBOX_INV[x] for x in data]

def rotate(data):
    if False:
        i = 10
        return i + 15
    return data[1:] + [data[0]]

def key_schedule_core(data, rcon_iteration):
    if False:
        while True:
            i = 10
    data = rotate(data)
    data = sub_bytes(data)
    data[0] = data[0] ^ RCON[rcon_iteration]
    return data

def xor(data1, data2):
    if False:
        for i in range(10):
            print('nop')
    return [x ^ y for (x, y) in zip(data1, data2)]

def iter_mix_columns(data, matrix):
    if False:
        print('Hello World!')
    for i in (0, 4, 8, 12):
        for row in matrix:
            mixed = 0
            for j in range(4):
                mixed ^= 0 if data[i:i + 4][j] == 0 or row[j] == 0 else RIJNDAEL_EXP_TABLE[(RIJNDAEL_LOG_TABLE[data[i + j]] + RIJNDAEL_LOG_TABLE[row[j]]) % 255]
            yield mixed

def shift_rows(data):
    if False:
        print('Hello World!')
    return [data[(column + row & 3) * 4 + row] for column in range(4) for row in range(4)]

def shift_rows_inv(data):
    if False:
        print('Hello World!')
    return [data[(column - row & 3) * 4 + row] for column in range(4) for row in range(4)]

def shift_block(data):
    if False:
        while True:
            i = 10
    data_shifted = []
    bit = 0
    for n in data:
        if bit:
            n |= 256
        bit = n & 1
        n >>= 1
        data_shifted.append(n)
    return data_shifted

def inc(data):
    if False:
        for i in range(10):
            print('nop')
    data = data[:]
    for i in range(len(data) - 1, -1, -1):
        if data[i] == 255:
            data[i] = 0
        else:
            data[i] = data[i] + 1
            break
    return data

def block_product(block_x, block_y):
    if False:
        i = 10
        return i + 15
    if len(block_x) != BLOCK_SIZE_BYTES or len(block_y) != BLOCK_SIZE_BYTES:
        raise ValueError('Length of blocks need to be %d bytes' % BLOCK_SIZE_BYTES)
    block_r = [225] + [0] * (BLOCK_SIZE_BYTES - 1)
    block_v = block_y[:]
    block_z = [0] * BLOCK_SIZE_BYTES
    for i in block_x:
        for bit in range(7, -1, -1):
            if i & 1 << bit:
                block_z = xor(block_z, block_v)
            do_xor = block_v[-1] & 1
            block_v = shift_block(block_v)
            if do_xor:
                block_v = xor(block_v, block_r)
    return block_z

def ghash(subkey, data):
    if False:
        print('Hello World!')
    if len(data) % BLOCK_SIZE_BYTES:
        raise ValueError('Length of data should be %d bytes' % BLOCK_SIZE_BYTES)
    last_y = [0] * BLOCK_SIZE_BYTES
    for i in range(0, len(data), BLOCK_SIZE_BYTES):
        block = data[i:i + BLOCK_SIZE_BYTES]
        last_y = block_product(xor(last_y, block), subkey)
    return last_y
__all__ = ['aes_cbc_decrypt', 'aes_cbc_decrypt_bytes', 'aes_ctr_decrypt', 'aes_decrypt_text', 'aes_decrypt', 'aes_ecb_decrypt', 'aes_gcm_decrypt_and_verify', 'aes_gcm_decrypt_and_verify_bytes', 'aes_cbc_encrypt', 'aes_cbc_encrypt_bytes', 'aes_ctr_encrypt', 'aes_ecb_encrypt', 'aes_encrypt', 'key_expansion', 'pad_block', 'pkcs7_padding', 'unpad_pkcs7']