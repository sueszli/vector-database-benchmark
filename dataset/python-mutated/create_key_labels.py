from google.cloud import kms

def create_key_labels(project_id: str, location_id: str, key_ring_id: str, key_id: str) -> kms.CryptoKey:
    if False:
        i = 10
        return i + 15
    "\n    Creates a new key in Cloud KMS with labels.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to create (e.g. 'my-labeled-key').\n\n    Returns:\n        CryptoKey: Cloud KMS key.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_ring_name = client.key_ring_path(project_id, location_id, key_ring_id)
    purpose = kms.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    algorithm = kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    key = {'purpose': purpose, 'version_template': {'algorithm': algorithm}, 'labels': {'team': 'alpha', 'cost_center': 'cc1234'}}
    created_key = client.create_crypto_key(request={'parent': key_ring_name, 'crypto_key_id': key_id, 'crypto_key': key})
    print(f'Created labeled key: {created_key.name}')
    return created_key