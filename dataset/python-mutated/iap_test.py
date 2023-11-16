"""Test script for Identity-Aware Proxy code samples."""
import os
import time
import pytest
import make_iap_request
import validate_jwt
REFLECT_SERVICE_HOSTNAME = 'gcp-devrel-iap-reflect.appspot.com'
IAP_CLIENT_ID = '320431926067-ldm6839p8l2sei41nlsfc632l4d0v2u1.apps.googleusercontent.com'
IAP_APP_ID = 'gcp-devrel-iap-reflect'
IAP_PROJECT_NUMBER = '320431926067'

@pytest.mark.flaky
def test_main(capsys):
    if False:
        print('Hello World!')
    if os.environ.get('TRAMPOLINE_CI', 'kokoro') != 'kokoro':
        pytest.skip('Only passing on Kokoro.')
    resp = make_iap_request.make_iap_request(f'https://{REFLECT_SERVICE_HOSTNAME}/', IAP_CLIENT_ID)
    iap_jwt = resp.split(': ').pop()
    expected_audience = '/projects/{}/apps/{}'.format(IAP_PROJECT_NUMBER, IAP_APP_ID)
    time.sleep(30)
    jwt_validation_result = validate_jwt.validate_iap_jwt(iap_jwt, expected_audience)
    assert not jwt_validation_result[2]
    assert jwt_validation_result[0]
    assert jwt_validation_result[1]