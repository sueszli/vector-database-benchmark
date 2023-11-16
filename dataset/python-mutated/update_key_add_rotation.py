import time
from google.cloud import kms

def update_key_add_rotation(project_id: str, location_id: str, key_ring_id: str, key_id: str) -> kms.CryptoKey:
    if False:
        i = 10
        return i + 15
    "\n    Add a rotation schedule to an existing key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n\n    Returns:\n        CryptoKey: Updated Cloud KMS key.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location_id, key_ring_id, key_id)
    key = {'name': key_name, 'rotation_period': {'seconds': 60 * 60 * 24 * 30}, 'next_rotation_time': {'seconds': int(time.time()) + 60 * 60 * 24}}
    update_mask = {'paths': ['rotation_period', 'next_rotation_time']}
    updated_key = client.update_crypto_key(request={'crypto_key': key, 'update_mask': update_mask})
    print(f'Updated key: {updated_key.name}')
    return updated_key