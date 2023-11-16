import argparse
from google.cloud import secretmanager

def update_secret_with_alias(project_id: str, secret_id: str) -> secretmanager.UpdateSecretRequest:
    if False:
        print('Hello World!')
    '\n    Update the metadata about an existing secret.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = client.secret_path(project_id, secret_id)
    secret = {'name': name, 'version_aliases': {'test': 1}}
    update_mask = {'paths': ['version_aliases']}
    response = client.update_secret(request={'secret': secret, 'update_mask': update_mask})
    print(f'Updated secret: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('--secret-id', required=True)
    args = parser.parse_args()
    update_secret_with_alias(args.project_id, args.secret_id)