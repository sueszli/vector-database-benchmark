from Cryptodome.Cipher import AES
from Cryptodome import Random
from . import CryptoMaterialsCacheEntry

def test_pycrypto():
    if False:
        for i in range(10):
            print('nop')
    key = b'Sixteen byte key'
    iv = Random.new().read(AES.block_size)
    cipher = pycrypto_arc2.new(key, AES.MODE_CFB, iv)
    factory = CryptoMaterialsCacheEntry()