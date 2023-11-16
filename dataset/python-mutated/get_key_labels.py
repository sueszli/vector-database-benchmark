from google.cloud import kms

def get_key_labels(project_id: str, location_id: str, key_ring_id: str, key_id: str) -> kms.CryptoKey:
    if False:
        while True:
            i = 10
    "\n    Get a key and its labels.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n\n    Returns:\n        CryptoKey: Cloud KMS key.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_name = client.crypto_key_path(project_id, location_id, key_ring_id, key_id)
    key = client.get_crypto_key(request={'name': key_name})
    for (k, v) in key.labels.items():
        print(f'{k} = {v}')
    return key