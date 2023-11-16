"""
command line application and sample code for listing secret versions of a
secret.
"""

def list_secret_versions_with_filter(project_id: str, secret_id: str, filter_str: str='state:ENABLED') -> None:
    if False:
        while True:
            i = 10
    '\n    List all secret versions in the given secret and their metadata.\n\n    Args:\n      project_id: Parent project id\n      secret_id: Parent secret id\n      filter_str: Secret version filter, constructing according to\n                  https://cloud.google.com/secret-manager/docs/filtering\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    parent = client.secret_path(project_id, secret_id)
    for version in client.list_secret_versions(request={'parent': parent, 'filter': filter_str}):
        print(f'Found secret version: {version.name}')