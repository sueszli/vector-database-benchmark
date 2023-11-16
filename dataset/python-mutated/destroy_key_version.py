from google.cloud import kms

def destroy_key_version(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str) -> kms.CryptoKeyVersion:
    if False:
        while True:
            i = 10
    "\n    Schedule destruction of the given key version.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the key version to destroy (e.g. '1').\n\n    Returns:\n        CryptoKeyVersion: The version.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    destroyed_version = client.destroy_crypto_key_version(request={'name': key_version_name})
    print(f'Destroyed key version: {destroyed_version.name}')
    return destroyed_version