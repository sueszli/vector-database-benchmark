"""
command line application and sample code for deleting an existing secret.
"""
import argparse

def delete_secret_with_etag(project_id: str, secret_id: str, etag: str) -> None:
    if False:
        return 10
    '\n    Delete the secret with the given name, etag, and all of its versions.\n    '
    from google.cloud import secretmanager
    from google.cloud.secretmanager_v1.types import service
    client = secretmanager.SecretManagerServiceClient()
    name = client.secret_path(project_id, secret_id)
    request = service.DeleteSecretRequest()
    request.name = name
    request.etag = etag
    client.delete_secret(request=request)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret to delete')
    parser.add_argument('etag', help='current etag of the secret to delete')
    args = parser.parse_args()
    delete_secret_with_etag(args.project_id, args.secret_id, args.etag)