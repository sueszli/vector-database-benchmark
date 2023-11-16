"""Sample showing how to validate the Identity-Aware Proxy (IAP) JWT.

This code should be used by applications in Google Compute Engine-based
environments (such as Google App Engine flexible environment, Google
Compute Engine, Google Kubernetes Engine, Google App Engine) to provide
an extra layer of assurance that a request was authorized by IAP.
"""
from google.auth.transport import requests
from google.oauth2 import id_token

def validate_iap_jwt(iap_jwt, expected_audience):
    if False:
        i = 10
        return i + 15
    'Validate an IAP JWT.\n\n    Args:\n      iap_jwt: The contents of the X-Goog-IAP-JWT-Assertion header.\n      expected_audience: The Signed Header JWT audience. See\n          https://cloud.google.com/iap/docs/signed-headers-howto\n          for details on how to get this value.\n\n    Returns:\n      (user_id, user_email, error_str).\n    '
    try:
        decoded_jwt = id_token.verify_token(iap_jwt, requests.Request(), audience=expected_audience, certs_url='https://www.gstatic.com/iap/verify/public_key')
        return (decoded_jwt['sub'], decoded_jwt['email'], '')
    except Exception as e:
        return (None, None, f'**ERROR: JWT validation error {e}**')