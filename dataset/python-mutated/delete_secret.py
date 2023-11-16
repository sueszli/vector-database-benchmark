"""
command line application and sample code for deleting an existing secret.
"""
import argparse

def delete_secret(project_id: str, secret_id: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Delete the secret with the given name and all of its versions.\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    name = client.secret_path(project_id, secret_id)
    client.delete_secret(request={'name': name})
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret to delete')
    args = parser.parse_args()
    delete_secret(args.project_id, args.secret_id)