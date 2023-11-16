"""
command line application and sample code for enabling a secret version.
"""
import argparse
from google.cloud import secretmanager

def enable_secret_version_with_etag(project_id: str, secret_id: str, version_id: str, etag: str) -> secretmanager.EnableSecretVersionRequest:
    if False:
        print('Hello World!')
    '\n    Enable the given secret version, enabling it to be accessed after\n    previously being disabled. Other secrets versions are unaffected.\n    '
    from google.cloud import secretmanager
    from google.cloud.secretmanager_v1.types import service
    client = secretmanager.SecretManagerServiceClient()
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    request = service.EnableSecretVersionRequest()
    request.name = name
    request.etag = etag
    response = client.enable_secret_version(request=request)
    print(f'Enabled secret version: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='id of the GCP project')
    parser.add_argument('secret_id', help='id of the secret from which to act')
    parser.add_argument('version_id', help='id of the version to enable')
    parser.add_argument('etag', help='current etag of the version')
    args = parser.parse_args()
    enable_secret_version_with_etag(args.project_id, args.secret_id, args.version_id, args.etag)