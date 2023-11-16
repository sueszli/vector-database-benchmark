from caller import request_primary_key_cipher_text, request_data_key_cipher_text, request_data_key_plaintext

def read_encrypted_key_file(encrypted_key_path):
    if False:
        return 10
    with open(encrypted_key_path, 'r') as file:
        original = file.readlines()
    return original[0]

def write_encrypted_key_file(encrypted_key_path, content):
    if False:
        while True:
            i = 10
    with open(encrypted_key_path, 'w') as file:
        file.write(content)

def retrieve_data_key_plaintext(ip, port, encrypted_primary_key_path, encrypted_data_key_path):
    if False:
        for i in range(10):
            print('nop')
    encrypted_primary_key = read_encrypted_key_file(encrypted_primary_key_path)
    encrypted_data_key = read_encrypted_key_file(encrypted_data_key_path)
    data_key_plaintext = request_data_key_plaintext(ip, port, encrypted_primary_key, encrypted_data_key)
    return data_key_plaintext

def generate_primary_key_cipher_text(ip, port):
    if False:
        return 10
    primary_key_cipher_text = request_primary_key_cipher_text(ip, port)
    write_encrypted_key_file('./encrypted_primary_key', primary_key_cipher_text)
    print('[INFO] Primary Key Generated Successfully at ./encrypted_primary_key')

def generate_data_key_cipher_text(ip, port, encrypted_primary_key_path, data_key_length=32):
    if False:
        i = 10
        return i + 15
    encrypted_primary_key = read_encrypted_key_file(encrypted_primary_key_path)
    data_key_cipher_text = request_data_key_cipher_text(ip, port, encrypted_primary_key, data_key_length)
    write_encrypted_key_file('./encrypted_data_key', data_key_cipher_text)
    print('[INFO] Data Key Generated Successfully at ./encrypted_data_key')