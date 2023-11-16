"""
command line application and sample code for adding a secret version with the
specified payload to an existing secret.
"""
import argparse
from google.cloud import secretmanager
import google_crc32c

def add_secret_version(project_id: str, secret_id: str, payload: str) -> secretmanager.SecretVersion:
    if False:
        for i in range(10):
            print('nop')
    '\n    Add a new secret version to the given secret with the provided payload.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    parent = client.secret_path(project_id, secret_id)
    payload_bytes = payload.encode('UTF-8')
    crc32c = google_crc32c.Checksum()
    crc32c.update(payload_bytes)
    response = client.add_secret_version(request={'parent': parent, 'payload': {'data': payload_bytes, 'data_crc32c': int(crc32c.hexdigest(), 16)}})
    print(f'Added secret version: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret in which to add')
    parser.add_argument('payload', help='secret material payload')
    args = parser.parse_args()
    add_secret_version(args.project_id, args.secret_id, args.payload)