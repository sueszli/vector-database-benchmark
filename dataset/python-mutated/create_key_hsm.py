import datetime
from google.cloud import kms
from google.protobuf import duration_pb2

def create_key_hsm(project_id: str, location_id: str, key_ring_id: str, key_id: str) -> kms.CryptoKey:
    if False:
        print('Hello World!')
    "\n    Creates a new key in Cloud KMS backed by Cloud HSM.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to create (e.g. 'my-hsm-key').\n\n    Returns:\n        CryptoKey: Cloud KMS key.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_ring_name = client.key_ring_path(project_id, location_id, key_ring_id)
    purpose = kms.CryptoKey.CryptoKeyPurpose.ENCRYPT_DECRYPT
    algorithm = kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.GOOGLE_SYMMETRIC_ENCRYPTION
    protection_level = kms.ProtectionLevel.HSM
    key = {'purpose': purpose, 'version_template': {'algorithm': algorithm, 'protection_level': protection_level}, 'destroy_scheduled_duration': duration_pb2.Duration().FromTimedelta(datetime.timedelta(days=1))}
    created_key = client.create_crypto_key(request={'parent': key_ring_name, 'crypto_key_id': key_id, 'crypto_key': key})
    print(f'Created hsm key: {created_key.name}')
    return created_key