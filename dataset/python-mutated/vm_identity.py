"""
Example of verifying Google Compute Engine virtual machine identity.

This sample will work only on a GCE virtual machine, as it relies on
communication with metadata server (https://cloud.google.com/compute/docs/storing-retrieving-metadata).

Example is used on: https://cloud.google.com/compute/docs/instances/verifying-instance-identity
"""
import pprint
import google.auth.transport.requests
from google.oauth2 import id_token
import requests
AUDIENCE_URL = 'http://www.example.com'
METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
METADATA_VM_IDENTITY_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={audience}&format={format}&licenses={licenses}'
FORMAT = 'full'
LICENSES = 'TRUE'

def acquire_token(audience: str=AUDIENCE_URL, format: str='standard', licenses: bool=True) -> str:
    if False:
        while True:
            i = 10
    "\n    Requests identity information from the metadata server.\n\n    Args:\n        audience: the unique URI agreed upon by both the instance and the\n            system verifying the instance's identity. For example, the audience\n            could be a URL for the connection between the two systems.\n        format: the optional parameter that specifies whether the project and\n            instance details are included in the payload. Specify `full` to\n            include this information in the payload or standard to omit the\n            information from the payload. The default value is `standard`.\n        licenses: an optional parameter that specifies whether license\n            codes for images associated with this instance are included in the\n            payload. Specify TRUE to include this information or FALSE to omit\n            this information from the payload. The default value is FALSE.\n            Has no effect unless format is `full`.\n\n    Returns:\n        A JSON Web Token signed using the RS256 algorithm. The token includes a\n        Google signature and additional information in the payload. You can send\n        this token to other systems and applications so that they can verify the\n        token and confirm that the identity of your instance.\n    "
    url = METADATA_VM_IDENTITY_URL.format(audience=audience, format=format, licenses=licenses)
    r = requests.get(url, headers=METADATA_HEADERS)
    r.raise_for_status()
    return r.text

def verify_token(token: str, audience: str) -> dict:
    if False:
        i = 10
        return i + 15
    "\n    Verify token signature and return the token payload.\n\n    Args:\n        token: the JSON Web Token received from the metadata server to\n            be verified.\n        audience: the unique URI agreed upon by both the instance and the\n            system verifying the instance's identity.\n\n    Returns:\n        Dictionary containing the token payload.\n    "
    request = google.auth.transport.requests.Request()
    payload = id_token.verify_token(token, request=request, audience=audience)
    return payload
if __name__ == '__main__':
    token_ = acquire_token(AUDIENCE_URL)
    print('Received token:', token_)
    print('Token verification:')
    pprint.pprint(verify_token(token_, AUDIENCE_URL))