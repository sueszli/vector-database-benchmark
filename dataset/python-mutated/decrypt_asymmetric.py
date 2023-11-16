from google.cloud import kms

def decrypt_asymmetric(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str, ciphertext: bytes) -> kms.DecryptResponse:
    if False:
        i = 10
        return i + 15
    "\n    Decrypt the ciphertext using an asymmetric key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the key version to use (e.g. '1').\n        ciphertext (bytes): Encrypted bytes to decrypt.\n\n    Returns:\n        DecryptResponse: Response including plaintext.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    ciphertext_crc32c = crc32c(ciphertext)
    decrypt_response = client.asymmetric_decrypt(request={'name': key_version_name, 'ciphertext': ciphertext, 'ciphertext_crc32c': ciphertext_crc32c})
    if not decrypt_response.verified_ciphertext_crc32c:
        raise Exception('The request sent to the server was corrupted in-transit.')
    if not decrypt_response.plaintext_crc32c == crc32c(decrypt_response.plaintext):
        raise Exception('The response received from the server was corrupted in-transit.')
    print(f'Plaintext: {decrypt_response.plaintext!r}')
    return decrypt_response

def crc32c(data: bytes) -> int:
    if False:
        return 10
    '\n    Calculates the CRC32C checksum of the provided data.\n    Args:\n        data: the bytes over which the checksum should be calculated.\n    Returns:\n        An int representing the CRC32C checksum of the provided bytes.\n    '
    import crcmod
    crc32c_fun = crcmod.predefined.mkPredefinedCrcFun('crc-32c')
    return crc32c_fun(data)