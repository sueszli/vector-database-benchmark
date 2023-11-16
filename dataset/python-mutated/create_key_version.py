from google.cloud import kms

def create_key_version(project_id: str, location_id: str, key_ring_id: str, key_id: str) -> kms.CryptoKey:
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates a new version of the given key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key for which to create a new version (e.g. 'my-key').\n\n    Returns:\n        CryptoKeyVersion: Cloud KMS key version.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location_id, key_ring_id, key_id)
    version = {}
    created_version = client.create_crypto_key_version(request={'parent': key_name, 'crypto_key_version': version})
    print(f'Created key version: {created_version.name}')
    return created_version