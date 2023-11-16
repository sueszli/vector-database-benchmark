import binascii
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
IKM = binascii.unhexlify(b'0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b')
L = 1200
OKM = HKDF(algorithm=hashes.SHA256(), length=L, salt=None, info=None).derive(IKM)

def _build_vectors():
    if False:
        i = 10
        return i + 15
    output = ['COUNT = 0', 'Hash = SHA-256', 'IKM = ' + binascii.hexlify(IKM).decode('ascii'), 'salt = ', 'info = ', f'L = {L}', 'OKM = ' + binascii.hexlify(OKM).decode('ascii')]
    return '\n'.join(output)

def _write_file(data, filename):
    if False:
        print('Hello World!')
    with open(filename, 'w') as f:
        f.write(data)
if __name__ == '__main__':
    _write_file(_build_vectors(), 'hkdf.txt')