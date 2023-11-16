"""
command line application and sample code for creating a new secret with
user managed replication.
"""
import argparse
import typing
from google.cloud import secretmanager

def create_ummr_secret(project_id: str, secret_id: str, locations: typing.List[str]) -> secretmanager.CreateSecretRequest:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a new secret with the given name. A secret is a logical wrapper\n    around a collection of secret versions. Secret versions hold the actual\n    secret material.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    parent = f'projects/{project_id}'
    response = client.create_secret(request={'parent': parent, 'secret_id': secret_id, 'secret': {'replication': {'user_managed': {'replicas': [{'location': x} for x in locations]}}}})
    print(f'Created secret: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret to create')
    parser.add_argument('--locations', nargs='+', help='list of locations for secret replication')
    args = parser.parse_args()
    create_ummr_secret(args.project_id, args.secret_id, args.locations)