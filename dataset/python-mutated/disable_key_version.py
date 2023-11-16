from google.cloud import kms

def disable_key_version(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str) -> kms.CryptoKeyVersion:
    if False:
        return 10
    "\n    Disable a key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the key version to disable (e.g. '1').\n\n    Returns:\n        CryptoKeyVersion: The version.\n\n    "
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    key_version = {'name': key_version_name, 'state': kms.CryptoKeyVersion.CryptoKeyVersionState.DISABLED}
    update_mask = {'paths': ['state']}
    disabled_version = client.update_crypto_key_version(request={'crypto_key_version': key_version, 'update_mask': update_mask})
    print(f'Disabled key version: {disabled_version.name}')
    return disabled_version