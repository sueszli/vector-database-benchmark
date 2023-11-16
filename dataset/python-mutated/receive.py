"""
Demonstrates how to receive authenticated service-to-service requests, eg
for Cloud Run or Cloud Functions
"""
from google.auth.transport import requests
from google.oauth2 import id_token

def receive_authorized_get_request(request):
    if False:
        print('Hello World!')
    "Parse the authorization header and decode the information\n    being sent by the Bearer token.\n\n    Args:\n        request: Flask request object\n\n    Returns:\n        The email from the request's Authorization header.\n    "
    auth_header = request.headers.get('Authorization')
    if auth_header:
        (auth_type, creds) = auth_header.split(' ', 1)
        if auth_type.lower() == 'bearer':
            claims = id_token.verify_token(creds, requests.Request())
            return f"Hello, {claims['email']}!\n"
        else:
            return f'Unhandled header format ({auth_type}).\n'
    return 'Hello, anonymous user.\n'