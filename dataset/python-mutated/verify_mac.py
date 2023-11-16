from google.cloud import kms

def verify_mac(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str, data: str, signature: bytes) -> kms.MacVerifyResponse:
    if False:
        print('Hello World!')
    "\n    Verify the signature of data from an HMAC key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): Version to use (e.g. '1').\n        data (string): Data that was signed.\n        signature (bytes): Signature bytes.\n\n    Returns:\n        MacVerifyResponse: Success.\n    "
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    data_bytes = data.encode('utf-8')
    verify_response = client.mac_verify(request={'name': key_version_name, 'data': data_bytes, 'mac': signature})
    print(f'Verified: {verify_response.success}')
    return verify_response