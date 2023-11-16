from pathlib import Path
from pprint import pprint
from typing import Union
from googleapiclient import discovery

def create_regional_certificate(project_id: str, region: str, certificate_file: Union[str, Path], private_key_file: Union[str, Path], certificate_name: str, description: str='Certificate created from a code sample.') -> dict:
    if False:
        return 10
    "\n    Create a regional SSL self-signed certificate within your Google Cloud project.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        region: name of the region you want to use.\n        certificate_file: path to the file with the certificate you want to create in your project.\n        private_key_file: path to the private key you used to sign the certificate with.\n        certificate_name: name for the certificate once it's created in your project.\n        description: description of the certificate.\n\n        Returns:\n        Dictionary with information about the new regional SSL self-signed certificate.\n    "
    service = discovery.build('compute', 'v1')
    with open(certificate_file) as f:
        _temp_cert = f.read()
    with open(private_key_file) as f:
        _temp_key = f.read()
    ssl_certificate_body = {'name': certificate_name, 'description': description, 'certificate': _temp_cert, 'privateKey': _temp_key}
    request = service.regionSslCertificates().insert(project=project_id, region=region, body=ssl_certificate_body)
    response = request.execute()
    pprint(response)
    return response