"""
command line application and sample code for accessing a secret version.
"""
import argparse
from google.cloud import secretmanager
import google_crc32c

def access_secret_version(project_id: str, secret_id: str, version_id: str) -> secretmanager.AccessSecretVersionResponse:
    if False:
        return 10
    '\n    Access the payload for the given secret version if one exists. The version\n    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = client.access_secret_version(request={'name': name})
    crc32c = google_crc32c.Checksum()
    crc32c.update(response.payload.data)
    if response.payload.data_crc32c != int(crc32c.hexdigest(), 16):
        print('Data corruption detected.')
        return response
    payload = response.payload.data.decode('UTF-8')
    print(f'Plaintext: {payload}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret to access')
    parser.add_argument('version_id', help='version to access')
    args = parser.parse_args()
    access_secret_version(args.project_id, args.secret_id, args.version_id)