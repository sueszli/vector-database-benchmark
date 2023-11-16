"""
command line application and sample code for listing secret versions of a
secret.
"""
import argparse

def list_secret_versions(project_id: str, secret_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    List all secret versions in the given secret and their metadata.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    parent = client.secret_path(project_id, secret_id)
    for version in client.list_secret_versions(request={'parent': parent}):
        print(f'Found secret version: {version.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret in which to list')
    args = parser.parse_args()
    list_secret_versions(args.project_id, args.secret_id)