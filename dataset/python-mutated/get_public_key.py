from google.cloud import kms

def get_public_key(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str) -> kms.PublicKey:
    if False:
        while True:
            i = 10
    "\n    Get the public key for an asymmetric key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the key to use (e.g. '1').\n\n    Returns:\n        PublicKey: Cloud KMS public key response.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    public_key = client.get_public_key(request={'name': key_version_name})
    if not public_key.name == key_version_name:
        raise Exception('The request sent to the server was corrupted in-transit.')
    if not public_key.pem_crc32c == crc32c(public_key.pem.encode('utf-8')):
        raise Exception('The response received from the server was corrupted in-transit.')
    print(f'Public key: {public_key.pem}')
    return public_key

def crc32c(data: bytes) -> int:
    if False:
        while True:
            i = 10
    '\n    Calculates the CRC32C checksum of the provided data.\n    Args:\n        data: the bytes over which the checksum should be calculated.\n    Returns:\n        An int representing the CRC32C checksum of the provided bytes.\n    '
    import crcmod
    crc32c_fun = crcmod.predefined.mkPredefinedCrcFun('crc-32c')
    return crc32c_fun(data)