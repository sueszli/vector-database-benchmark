from google.cloud import kms

def restore_key_version(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str) -> kms.CryptoKeyVersion:
    if False:
        print('Hello World!')
    "\n    Restore a key version scheduled for destruction.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the version to use (e.g. '1').\n\n    Returns:\n        CryptoKeyVersion: Restored Cloud KMS key version.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    restored_version = client.restore_crypto_key_version(request={'name': key_version_name})
    print(f'Restored key version: {restored_version.name}')
    return restored_version