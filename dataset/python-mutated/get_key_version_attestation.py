from google.cloud import kms

def get_key_version_attestation(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str) -> kms.KeyOperationAttestation:
    if False:
        i = 10
        return i + 15
    "\n    Get an HSM-backend key's attestation.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the version to use (e.g. '1').\n\n    Returns:\n        Attestation: Cloud KMS key attestation.\n\n    "
    import base64
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    version = client.get_crypto_key_version(request={'name': key_version_name})
    attestation = version.attestation
    if not attestation:
        raise 'no attestation - attestations only exist on HSM keys'
    encoded_attestation = base64.b64encode(attestation.content)
    print(f'Got key attestation: {encoded_attestation!r}')
    return attestation