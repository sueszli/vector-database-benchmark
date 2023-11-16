import hashlib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import utils
from google.cloud import kms

def verify_asymmetric_ec(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str, message: str, signature: str) -> bool:
    if False:
        while True:
            i = 10
    "\n    Verify the signature of an message signed with an asymmetric EC key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the version to use (e.g. '1').\n        message (string): Original message (e.g. 'my message')\n        signature (bytes): Signature from a sign request.\n\n    Returns:\n        bool: True if verified, False otherwise\n\n    "
    message_bytes = message.encode('utf-8')
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    public_key = client.get_public_key(request={'name': key_version_name})
    pem = public_key.pem.encode('utf-8')
    ec_key = serialization.load_pem_public_key(pem, default_backend())
    hash_ = hashlib.sha256(message_bytes).digest()
    try:
        sha256 = hashes.SHA256()
        ec_key.verify(signature, hash_, ec.ECDSA(utils.Prehashed(sha256)))
        print('Signature verified')
        return True
    except InvalidSignature:
        print('Signature failed to verify')
        return False