"""Example of calling a Google Cloud Endpoint API with an ID token obtained
using the Google OAuth2 flow."""
import argparse
import google_auth_oauthlib.flow
import requests
from six.moves import urllib

def get_id_token(client_secrets_file, extra_args):
    if False:
        i = 10
        return i + 15
    'Obtains credentials from the user using OAuth 2.0 and then returns the\n    ID token from those credentials.'
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes=['openid', 'email', 'profile'])
    flow.run_local_server()
    id_token = flow.oauth2session.token['id_token']
    return id_token

def make_request(host, api_key, id_token):
    if False:
        return 10
    'Makes a request to the auth info endpoint for Google ID tokens.'
    url = urllib.parse.urljoin(host, '/auth/info/googleidtoken')
    params = {'key': api_key}
    headers = {'Authorization': 'Bearer {}'.format(id_token)}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.text

def main(host, api_key, client_secrets_file, extra_args):
    if False:
        for i in range(10):
            print('nop')
    id_token = get_id_token(client_secrets_file, extra_args)
    response = make_request(host, api_key, id_token)
    print(response)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('host', help='Your API host, e.g. https://your-project.appspot.com.')
    parser.add_argument('api_key', help='Your API key.')
    parser.add_argument('client_secrets_file', help='The path to your OAuth2 client secrets file.')
    args = parser.parse_args()
    main(args.host, args.api_key, args.client_secrets_file, args)