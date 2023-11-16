"""Example of authenticating using access tokens directly on Compute Engine.

For more information, see the README.md under /compute.
"""
import argparse
import requests
METADATA_URL = 'http://metadata.google.internal/computeMetadata/v1/'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
SERVICE_ACCOUNT = 'default'

def get_access_token() -> str:
    if False:
        return 10
    '\n    Retrieves access token from the metadata server.\n\n    Returns:\n        The access token.\n    '
    url = f'{METADATA_URL}instance/service-accounts/{SERVICE_ACCOUNT}/token'
    r = requests.get(url, headers=METADATA_HEADERS)
    r.raise_for_status()
    access_token = r.json()['access_token']
    return access_token

def list_buckets(project_id: str, access_token: str) -> dict:
    if False:
        i = 10
        return i + 15
    '\n    Calls Storage API to retrieve a list of buckets.\n\n    Args:\n        project_id: name of the project to list buckets from.\n        access_token: access token to authenticate with.\n\n    Returns:\n        Response from the API.\n    '
    url = 'https://www.googleapis.com/storage/v1/b'
    params = {'project': project_id}
    headers = {'Authorization': f'Bearer {access_token}'}
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    return r.json()

def main(project_id: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Retrieves access token from metadata server and uses it to list\n    buckets in a project.\n\n    Args:\n        project_id: name of the project to list buckets from.\n    '
    access_token = get_access_token()
    buckets = list_buckets(project_id, access_token)
    print(buckets)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID.')
    args = parser.parse_args()
    main(args.project_id)