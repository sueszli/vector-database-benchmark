"""Example of generating a JWT signed from a service account file."""
import argparse
import json
import time
import google.auth.crypt
import google.auth.jwt
'Max lifetime of the token (one hour, in seconds).'
MAX_TOKEN_LIFETIME_SECS = 3600

def generate_jwt(service_account_file, issuer, audiences):
    if False:
        while True:
            i = 10
    'Generates a signed JSON Web Token using a Google API Service Account.'
    with open(service_account_file) as fh:
        service_account_info = json.load(fh)
    signer = google.auth.crypt.RSASigner.from_string(service_account_info['private_key'], service_account_info['private_key_id'])
    now = int(time.time())
    payload = {'iat': now, 'exp': now + MAX_TOKEN_LIFETIME_SECS, 'aud': audiences, 'iss': issuer, 'sub': issuer, 'email': 'user@example.com'}
    signed_jwt = google.auth.jwt.encode(signer, payload)
    return signed_jwt
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--file', help='The path to your service account json file.')
    parser.add_argument('--issuer', default='', help='issuer')
    parser.add_argument('--audiences', default='', help='audiences')
    args = parser.parse_args()
    signed_jwt = generate_jwt(args.file, args.issuer, args.audiences)
    print(signed_jwt.decode('utf-8'))