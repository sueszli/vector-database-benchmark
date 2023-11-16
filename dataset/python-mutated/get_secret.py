"""
command line application and sample code for getting metadata about a secret.
"""
import argparse
from google.cloud import secretmanager

def get_secret(project_id: str, secret_id: str) -> secretmanager.GetSecretRequest:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get information about the given secret. This only returns metadata about\n    the secret container, not any secret material.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = client.secret_path(project_id, secret_id)
    response = client.get_secret(request={'name': name})
    if 'automatic' in response.replication:
        replication = 'AUTOMATIC'
    elif 'user_managed' in response.replication:
        replication = 'MANAGED'
    else:
        raise Exception(f'Unknown replication {response.replication}')
    print(f'Got secret {response.name} with replication policy {replication}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret to get')
    args = parser.parse_args()
    get_secret(args.project_id, args.secret_id)