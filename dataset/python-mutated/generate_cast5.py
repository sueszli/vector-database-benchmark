import binascii
from cryptography.hazmat.primitives.ciphers import algorithms, base, modes

def encrypt(mode, key, iv, plaintext):
    if False:
        print('Hello World!')
    cipher = base.Cipher(algorithms.CAST5(binascii.unhexlify(key)), mode(binascii.unhexlify(iv)))
    encryptor = cipher.encryptor()
    ct = encryptor.update(binascii.unhexlify(plaintext))
    ct += encryptor.finalize()
    return binascii.hexlify(ct)

def build_vectors(mode, filename):
    if False:
        while True:
            i = 10
    count = 0
    output = []
    key = None
    iv = None
    plaintext = None
    with open(filename) as vector_file:
        for line in vector_file:
            line = line.strip()
            if line.startswith('KEY'):
                if count != 0:
                    output.append(f'CIPHERTEXT = {encrypt(mode, key, iv, plaintext)}')
                output.append(f'\nCOUNT = {count}')
                count += 1
                (_, key) = line.split(' = ')
                output.append(f'KEY = {key}')
            elif line.startswith('IV'):
                (_, iv) = line.split(' = ')
                iv = iv[0:16]
                output.append(f'IV = {iv}')
            elif line.startswith('PLAINTEXT'):
                (_, plaintext) = line.split(' = ')
                output.append(f'PLAINTEXT = {plaintext}')
        output.append(f'CIPHERTEXT = {encrypt(mode, key, iv, plaintext)}')
    return '\n'.join(output)

def write_file(data, filename):
    if False:
        return 10
    with open(filename, 'w') as f:
        f.write(data)
cbc_path = 'tests/hazmat/primitives/vectors/ciphers/AES/CBC/CBCMMT128.rsp'
write_file(build_vectors(modes.CBC, cbc_path), 'cast5-cbc.txt')
ofb_path = 'tests/hazmat/primitives/vectors/ciphers/AES/OFB/OFBMMT128.rsp'
write_file(build_vectors(modes.OFB, ofb_path), 'cast5-ofb.txt')
cfb_path = 'tests/hazmat/primitives/vectors/ciphers/AES/CFB/CFB128MMT128.rsp'
write_file(build_vectors(modes.CFB, cfb_path), 'cast5-cfb.txt')
ctr_path = 'tests/hazmat/primitives/vectors/ciphers/AES/CTR/aes-128-ctr.txt'
write_file(build_vectors(modes.CTR, ctr_path), 'cast5-ctr.txt')