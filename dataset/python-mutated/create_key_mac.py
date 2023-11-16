import datetime
from google.cloud import kms
from google.protobuf import duration_pb2

def create_key_mac(project_id: str, location_id: str, key_ring_id: str, key_id: str) -> kms.CryptoKey:
    if False:
        return 10
    "\n    Creates a new key in Cloud KMS for HMAC operations.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to create (e.g. 'my-mac-key').\n\n    Returns:\n        CryptoKey: Cloud KMS key.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_ring_name = client.key_ring_path(project_id, location_id, key_ring_id)
    purpose = kms.CryptoKey.CryptoKeyPurpose.MAC
    algorithm = kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.HMAC_SHA256
    key = {'purpose': purpose, 'version_template': {'algorithm': algorithm}, 'destroy_scheduled_duration': duration_pb2.Duration().FromTimedelta(datetime.timedelta(days=1))}
    created_key = client.create_crypto_key(request={'parent': key_ring_name, 'crypto_key_id': key_id, 'crypto_key': key})
    print(f'Created mac key: {created_key.name}')
    return created_key