import base64
from google.cloud import kms

def generate_random_bytes(project_id: str, location_id: str, num_bytes: int) -> bytes:
    if False:
        print('Hello World!')
    "\n    Generate random bytes with entropy sourced from the given location.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        num_bytes (integer): number of bytes of random data.\n\n    Returns:\n        bytes: Encrypted ciphertext.\n\n    "
    client = kms.KeyManagementServiceClient()
    location_name = client.common_location_path(project_id, location_id)
    protection_level = kms.ProtectionLevel.HSM
    random_bytes_response = client.generate_random_bytes(request={'location': location_name, 'length_bytes': num_bytes, 'protection_level': protection_level})
    print(f'Random bytes: {base64.b64encode(random_bytes_response.data)}')
    return random_bytes_response