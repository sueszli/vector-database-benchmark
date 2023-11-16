"""Example of authenticating using Application Default Credentials on
Compute Engine.

For more information, see the README.md under /compute.
"""
import argparse
from typing import List
from google.cloud import storage

def create_client() -> storage.Client:
    if False:
        print('Hello World!')
    '\n    Construct a client object for the Storage API using the\n    application default credentials.\n\n    Returns:\n        Storage API client object.\n    '
    return storage.Client()

def list_buckets(client: storage.Client, project_id: str) -> List[storage.Bucket]:
    if False:
        i = 10
        return i + 15
    '\n    Retrieve bucket list of a project using provided client object.\n\n\n    Args:\n        client: Storage API client object.\n        project_id: name of the project to list buckets from.\n\n    Returns:\n        List of Buckets found in the project.\n    '
    buckets = client.list_buckets()
    return list(buckets)

def main(project_id: str) -> None:
    if False:
        while True:
            i = 10
    client = create_client()
    buckets = list_buckets(client, project_id)
    print(buckets)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud Project ID.')
    args = parser.parse_args()
    main(args.project_id)