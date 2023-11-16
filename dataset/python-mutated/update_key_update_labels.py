from google.cloud import kms

def update_key_update_labels(project_id: str, location_id: str, key_ring_id: str, key_id: str) -> kms.CryptoKey:
    if False:
        return 10
    "\n    Update labels on an existing key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n\n    Returns:\n        CryptoKey: Updated Cloud KMS key.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location_id, key_ring_id, key_id)
    key = {'name': key_name, 'labels': {'new_label': 'new_value'}}
    update_mask = {'paths': ['labels']}
    updated_key = client.update_crypto_key(request={'crypto_key': key, 'update_mask': update_mask})
    print(f'Updated key: {updated_key.name}')
    return updated_key