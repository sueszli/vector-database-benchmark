import base64
import hashlib
from google.cloud import kms

def sign_asymmetric(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str, message: str) -> kms.AsymmetricSignResponse:
    if False:
        for i in range(10):
            print('nop')
    "\n    Sign a message using the private key part of an asymmetric key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): Version to use (e.g. '1').\n        message (string): Message to sign.\n\n    Returns:\n        AsymmetricSignResponse: Signature.\n    "
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    message_bytes = message.encode('utf-8')
    hash_ = hashlib.sha256(message_bytes).digest()
    digest = {'sha256': hash_}
    digest_crc32c = crc32c(hash_)
    sign_response = client.asymmetric_sign(request={'name': key_version_name, 'digest': digest, 'digest_crc32c': digest_crc32c})
    if not sign_response.verified_digest_crc32c:
        raise Exception('The request sent to the server was corrupted in-transit.')
    if not sign_response.name == key_version_name:
        raise Exception('The request sent to the server was corrupted in-transit.')
    if not sign_response.signature_crc32c == crc32c(sign_response.signature):
        raise Exception('The response received from the server was corrupted in-transit.')
    print(f'Signature: {base64.b64encode(sign_response.signature)!r}')
    return sign_response

def crc32c(data: bytes) -> int:
    if False:
        print('Hello World!')
    '\n    Calculates the CRC32C checksum of the provided data.\n    Args:\n        data: the bytes over which the checksum should be calculated.\n    Returns:\n        An int representing the CRC32C checksum of the provided bytes.\n    '
    import crcmod
    crc32c_fun = crcmod.predefined.mkPredefinedCrcFun('crc-32c')
    return crc32c_fun(data)