"""Example of calling a Google Cloud Endpoint API with a JWT signed by
a Google API Service Account."""
import argparse
import time
import google.auth.crypt
import google.auth.jwt
import requests

def generate_jwt(sa_keyfile, sa_email='account@project-id.iam.gserviceaccount.com', audience='your-service-name', expiry_length=3600):
    if False:
        i = 10
        return i + 15
    'Generates a signed JSON Web Token using a Google API Service Account.'
    now = int(time.time())
    payload = {'iat': now, 'exp': now + expiry_length, 'iss': sa_email, 'aud': audience, 'sub': sa_email, 'email': sa_email}
    signer = google.auth.crypt.RSASigner.from_service_account_file(sa_keyfile)
    jwt = google.auth.jwt.encode(signer, payload)
    return jwt

def make_jwt_request(signed_jwt, url='https://your-endpoint.com'):
    if False:
        print('Hello World!')
    'Makes an authorized request to the endpoint'
    headers = {'Authorization': 'Bearer {}'.format(signed_jwt.decode('utf-8')), 'content-type': 'application/json'}
    response = requests.get(url, headers=headers)
    print(response.status_code, response.content)
    response.raise_for_status()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('host', help='Your API host, e.g. https://your-project.appspot.com.')
    parser.add_argument('audience', help='The aud entry for the JWT')
    parser.add_argument('sa_path', help='The path to your service account json file.')
    parser.add_argument('sa_email', help='The email address for the service account.')
    args = parser.parse_args()
    expiry_length = 3600
    keyfile_jwt = generate_jwt(args.sa_path, args.sa_email, args.audience, expiry_length)
    make_jwt_request(keyfile_jwt, args.host)