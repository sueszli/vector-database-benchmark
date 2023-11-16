from google.cloud import kms

def create_key_ring(project_id: str, location_id: str, key_ring_id: str) -> kms.CryptoKey:
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates a new key ring in Cloud KMS\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the key ring to create (e.g. 'my-key-ring').\n\n    Returns:\n        KeyRing: Cloud KMS key ring.\n\n    "
    client = kms.KeyManagementServiceClient()
    location_name = f'projects/{project_id}/locations/{location_id}'
    key_ring = {}
    created_key_ring = client.create_key_ring(request={'parent': location_name, 'key_ring_id': key_ring_id, 'key_ring': key_ring})
    print(f'Created key ring: {created_key_ring.name}')
    return created_key_ring