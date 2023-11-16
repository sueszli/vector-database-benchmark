from google.cloud import kms

def sign_mac(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str, data: str) -> kms.MacSignResponse:
    if False:
        return 10
    "\n    Sign a message using the private key part of an asymmetric key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): Version to use (e.g. '1').\n        data (string): Data to sign.\n\n    Returns:\n        MacSignResponse: Signature.\n    "
    import base64
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    data_bytes = data.encode('utf-8')
    sign_response = client.mac_sign(request={'name': key_version_name, 'data': data_bytes})
    print(f'Signature: {base64.b64encode(sign_response.mac)!r}')
    return sign_response