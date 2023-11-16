"""
command line application and sample code for listing secrets in a project.
"""

def list_secrets_with_filter(project_id: str, filter_str: str) -> None:
    if False:
        return 10
    '\n    List all secrets in the given project.\n\n    Args:\n      project_id: Parent project id\n      filter_str: Secret filter, constructing according to\n                  https://cloud.google.com/secret-manager/docs/filtering\n    '
    from google.cloud import secretmanager
    client = secretmanager.SecretManagerServiceClient()
    parent = f'projects/{project_id}'
    for secret in client.list_secrets(request={'parent': parent, 'filter': filter_str}):
        print(f'Found secret: {secret.name}')