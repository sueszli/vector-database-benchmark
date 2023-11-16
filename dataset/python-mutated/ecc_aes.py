from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.PublicKey import ECC
from Crypto.Util.Padding import pad, unpad

class ECC_AES:

    def __init__(self, curve='P-256'):
        if False:
            i = 10
            return i + 15
        self.curve = curve
        self._aes_key = None
        self._private_key = None

    def _derive_aes_key(self, ecc_key, nonce):
        if False:
            return 10
        if not self._aes_key:
            hasher = SHA256.new()
            hasher.update(ecc_key.export_key(format='DER') + nonce.encode())
            self._aes_key = hasher.digest()[:32]
        return self._aes_key

    def generate_key_pair(self):
        if False:
            i = 10
            return i + 15
        private_key = ECC.generate(curve=self.curve)
        public_key = private_key.public_key()
        pem_private = private_key.export_key(format='PEM')
        pem_public = public_key.export_key(format='PEM')
        return (pem_private, pem_public)

    def load_private_key(self, private_key_pem):
        if False:
            while True:
                i = 10
        self._private_key = ECC.import_key(private_key_pem)
        self._aes_key = None

    def encrypt(self, text, nonce):
        if False:
            print('Hello World!')
        if not self._private_key:
            raise ValueError('Private key not loaded')
        aes_key = self._derive_aes_key(self._private_key, nonce)
        cipher = AES.new(aes_key, AES.MODE_ECB)
        padded_text = pad(text.encode(), AES.block_size)
        ciphertext = cipher.encrypt(padded_text)
        return ciphertext

    def decrypt(self, ciphertext, nonce):
        if False:
            for i in range(10):
                print('nop')
        if not self._private_key:
            raise ValueError('Private key not loaded')
        aes_key = self._derive_aes_key(self._private_key, nonce)
        cipher = AES.new(aes_key, AES.MODE_ECB)
        padded_plaintext = cipher.decrypt(ciphertext)
        plaintext = unpad(padded_plaintext, AES.block_size)
        return plaintext.decode()
if __name__ == '__main__':
    ecc_aes = ECC_AES()
    (private_key, public_key) = ecc_aes.generate_key_pair()
    ecc_aes.load_private_key(private_key)
    nonce = 'THIS-IS-USER-ID'
    print(private_key)
    message = 'Hello, this is a secret message!'
    encrypted_message = ecc_aes.encrypt(message, nonce)
    print(f'Encrypted message: {encrypted_message.hex()}')
    decrypted_message = ecc_aes.decrypt(encrypted_message, nonce)
    print(f'Decrypted message: {decrypted_message}')
    assert message == decrypted_message, 'Original message and decrypted message do not match'