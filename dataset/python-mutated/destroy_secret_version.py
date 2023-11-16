"""
command line application and sample code for destroying a secret version.
"""
import argparse
from google.cloud import secretmanager

def destroy_secret_version(project_id: str, secret_id: str, version_id: str) -> secretmanager.DestroySecretVersionRequest:
    if False:
        print('Hello World!')
    '\n    Destroy the given secret version, making the payload irrecoverable. Other\n    secrets versions are unaffected.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = client.destroy_secret_version(request={'name': name})
    print(f'Destroyed secret version: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret from which to act')
    parser.add_argument('version_id', help='id of the version to destroy')
    args = parser.parse_args()
    destroy_secret_version(args.project_id, args.secret_id, args.version_id)