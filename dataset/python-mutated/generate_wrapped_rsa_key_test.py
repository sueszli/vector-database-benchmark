import os
import uuid
import googleapiclient.discovery
import generate_wrapped_rsa_key
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_main() -> None:
    if False:
        for i in range(10):
            print('nop')
    generate_wrapped_rsa_key.main(None)

def test_create_disk() -> None:
    if False:
        print('Hello World!')
    compute = googleapiclient.discovery.build('compute', 'beta')
    key_bytes = os.urandom(32)
    google_public_key = generate_wrapped_rsa_key.get_google_public_cert_key()
    wrapped_rsa_key = generate_wrapped_rsa_key.wrap_rsa_key(google_public_key, key_bytes)
    disk_name = f'new-encrypted-disk-{uuid.uuid4().hex}'
    try:
        compute.disks().insert(project=PROJECT, zone='us-central1-f', body={'name': disk_name, 'diskEncryptionKey': {'rsaEncryptedKey': wrapped_rsa_key.decode('utf-8')}}).execute()
    finally:
        compute.disks().delete(project=PROJECT, zone='us-central1-f', disk=disk_name).execute()