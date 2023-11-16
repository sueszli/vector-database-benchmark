import base64
from google.cloud import kms

def encrypt_symmetric(project_id: str, location_id: str, key_ring_id: str, key_id: str, plaintext: str) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    "\n    Encrypt plaintext using a symmetric key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        plaintext (string): message to encrypt\n\n    Returns:\n        bytes: Encrypted ciphertext.\n\n    "
    plaintext_bytes = plaintext.encode('utf-8')
    plaintext_crc32c = crc32c(plaintext_bytes)
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location_id, key_ring_id, key_id)
    encrypt_response = client.encrypt(request={'name': key_name, 'plaintext': plaintext_bytes, 'plaintext_crc32c': plaintext_crc32c})
    if not encrypt_response.verified_plaintext_crc32c:
        raise Exception('The request sent to the server was corrupted in-transit.')
    if not encrypt_response.ciphertext_crc32c == crc32c(encrypt_response.ciphertext):
        raise Exception('The response received from the server was corrupted in-transit.')
    print(f'Ciphertext: {base64.b64encode(encrypt_response.ciphertext)}')
    return encrypt_response

def crc32c(data: bytes) -> int:
    if False:
        while True:
            i = 10
    '\n    Calculates the CRC32C checksum of the provided data.\n\n    Args:\n        data: the bytes over which the checksum should be calculated.\n\n    Returns:\n        An int representing the CRC32C checksum of the provided bytes.\n    '
    import crcmod
    crc32c_fun = crcmod.predefined.mkPredefinedCrcFun('crc-32c')
    return crc32c_fun(data)