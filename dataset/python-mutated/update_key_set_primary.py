from google.cloud import kms

def update_key_set_primary(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str) -> kms.CryptoKey:
    if False:
        print('Hello World!')
    "\n    Update the primary version of a key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the key to make primary (e.g. '2').\n\n    Returns:\n        CryptoKey: Updated Cloud KMS key.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location_id, key_ring_id, key_id)
    updated_key = client.update_crypto_key_primary_version(request={'name': key_name, 'crypto_key_version_id': version_id})
    print(f'Updated {updated_key.name} primary to {version_id}')
    return updated_key