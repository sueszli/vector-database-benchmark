import os
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import keywrap
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from google.cloud import kms

def import_manually_wrapped_key(project_id: str, location_id: str, key_ring_id: str, crypto_key_id: str, import_job_id: str) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Generates and imports local key material to Cloud KMS.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        crypto_key_id (string): ID of the key to import (e.g. 'my-asymmetric-signing-key').\n        import_job_id (string): ID of the import job (e.g. 'my-import-job').\n    "
    key = ec.generate_private_key(ec.SECP256R1, backends.default_backend())
    formatted_key = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    print(f'Generated key bytes: {formatted_key!r}')
    client = kms.KeyManagementServiceClient()
    crypto_key_name = client.crypto_key_path(project_id, location_id, key_ring_id, crypto_key_id)
    import_job_name = client.import_job_path(project_id, location_id, key_ring_id, import_job_id)
    kwp_key = os.urandom(32)
    wrapped_target_key = keywrap.aes_key_wrap_with_padding(kwp_key, formatted_key, backends.default_backend())
    import_job = client.get_import_job(name=import_job_name)
    import_job_pub = serialization.load_pem_public_key(bytes(import_job.public_key.pem, 'UTF-8'), backends.default_backend())
    wrapped_kwp_key = import_job_pub.encrypt(kwp_key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA1()), algorithm=hashes.SHA1(), label=None))
    client.import_crypto_key_version({'parent': crypto_key_name, 'import_job': import_job_name, 'algorithm': kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm.EC_SIGN_P256_SHA256, 'rsa_aes_wrapped_key': wrapped_kwp_key + wrapped_target_key})
    print(f'Imported: {import_job.name}')