import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from google.cloud import kms

def encrypt_asymmetric(project_id: str, location_id: str, key_ring_id: str, key_id: str, version_id: str, plaintext: str) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    "\n    Encrypt plaintext using the public key portion of an asymmetric key.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        key_id (string): ID of the key to use (e.g. 'my-key').\n        version_id (string): ID of the key version to use (e.g. '1').\n        plaintext (string): message to encrypt\n\n    Returns:\n        bytes: Encrypted ciphertext.\n\n    "
    plaintext_bytes = plaintext.encode('utf-8')
    client = kms.KeyManagementServiceClient()
    key_version_name = client.crypto_key_version_path(project_id, location_id, key_ring_id, key_id, version_id)
    public_key = client.get_public_key(request={'name': key_version_name})
    pem = public_key.pem.encode('utf-8')
    rsa_key = serialization.load_pem_public_key(pem, default_backend())
    sha256 = hashes.SHA256()
    mgf = padding.MGF1(algorithm=sha256)
    pad = padding.OAEP(mgf=mgf, algorithm=sha256, label=None)
    ciphertext = rsa_key.encrypt(plaintext_bytes, pad)
    print(f'Ciphertext: {base64.b64encode(ciphertext)!r}')
    return ciphertext