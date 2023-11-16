"""
command line application and sample code for listing secrets in a project.
"""
import argparse

def list_secrets(project_id: str) -> None:
    if False:
        while True:
            i = 10
    '\n    List all secrets in the given project.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    parent = f'projects/{project_id}'
    for secret in client.list_secrets(request={'parent': parent}):
        print(f'Found secret: {secret.name}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    args = parser.parse_args()
    list_secrets(args.project_id)