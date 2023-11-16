"""
command line application and sample code for getting metadata about a secret
version, but not the secret payload.
"""
import argparse
from google.cloud import secretmanager

def get_secret_version(project_id: str, secret_id: str, version_id: str) -> secretmanager.GetSecretVersionRequest:
    if False:
        while True:
            i = 10
    '\n    Get information about the given secret version. It does not include the\n    payload data.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = client.get_secret_version(request={'name': name})
    state = response.state.name
    print(f'Got secret version {response.name} with state {state}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret from which to act')
    parser.add_argument('version_id', help='id of the version to get')
    args = parser.parse_args()
    get_secret_version(args.project_id, args.secret_id, args.version_id)