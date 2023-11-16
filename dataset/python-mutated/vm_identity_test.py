import time
from google.auth import jwt
import pytest
import requests
import vm_identity
AUDIENCE = 'http://www.testing.com'

def wait_for_token(token):
    if False:
        i = 10
        return i + 15
    '\n    This function will wait if the Issued At value of the token is in the future.\n    It will not validate the token in any way.\n    '
    decoded = jwt.decode(token, verify=False)
    time.sleep(max(0, int(decoded['iat']) - int(time.time())))
    return

def test_vm_identity():
    if False:
        for i in range(10):
            print('nop')
    try:
        r = requests.get('http://metadata.google.internal/computeMetadata/v1/project/project-id', headers={'Metadata-Flavor': 'Google'})
        project_id = r.text
    except requests.exceptions.ConnectionError:
        pytest.skip('Test can only be run inside GCE VM.')
        return
    token = vm_identity.acquire_token(AUDIENCE, 'full', True)
    assert isinstance(token, str) and token
    wait_for_token(token)
    verification = vm_identity.verify_token(token, AUDIENCE)
    assert isinstance(verification, dict) and verification
    assert verification['aud'] == AUDIENCE
    assert verification['email_verified']
    assert verification['iss'] == 'https://accounts.google.com'
    assert verification['google']['compute_engine']['project_id'] == project_id