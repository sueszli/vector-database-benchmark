from cryptography.fernet import Fernet, InvalidToken, InvalidSignature
key = b'e3mp0E0Jr3jnVb96A31_lKzGZlSTPIp4-rPaVseyn58='
cipher_suite = Fernet(key)

def encrypt_data(data):
    if False:
        return 10
    '\n    Encrypts the given data using the Fernet cipher suite.\n\n    Args:\n        data (str): The data to be encrypted.\n\n    Returns:\n        str: The encrypted data, decoded as a string.\n    '
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data.decode()

def decrypt_data(encrypted_data):
    if False:
        i = 10
        return i + 15
    '\n    Decrypts the given encrypted data using the Fernet cipher suite.\n\n    Args:\n        encrypted_data (str): The encrypted data to be decrypted.\n\n    Returns:\n        str: The decrypted data, decoded as a string.\n    '
    decrypted_data = cipher_suite.decrypt(encrypted_data.encode())
    return decrypted_data.decode()

def is_encrypted(value):
    if False:
        for i in range(10):
            print('nop')
    key = b'e3mp0E0Jr3jnVb96A31_lKzGZlSTPIp4-rPaVseyn58='
    try:
        f = Fernet(key)
        f.decrypt(value)
        return True
    except (InvalidToken, InvalidSignature):
        return False
    except (ValueError, TypeError):
        return False