import base64
import unittest
from http import HTTPStatus
import requests
import requests_mock
from source_zoom.components import ServerToServerOauthAuthenticator

class TestOAuthClient(unittest.TestCase):

    def test_generate_access_token(self):
        if False:
            while True:
                i = 10
        except_access_token = 'rc-test-token'
        except_token_response = {'access_token': except_access_token}
        config = {'account_id': 'rc-asdfghjkl', 'client_id': 'rc-123456789', 'client_secret': 'rc-test-secret', 'authorization_endpoint': 'https://example.zoom.com/oauth/token', 'grant_type': 'account_credentials'}
        parameters = config
        client = ServerToServerOauthAuthenticator(config=config, account_id=config['account_id'], client_id=config['client_id'], client_secret=config['client_secret'], grant_type=config['grant_type'], authorization_endpoint=config['authorization_endpoint'], parameters=parameters)
        token = base64.b64encode(f"{config.get('client_id')}:{config.get('client_secret')}".encode('ascii')).decode('utf-8')
        headers = {'Authorization': f'Basic {token}', 'Content-type': 'application/json'}
        url = f"{config.get('authorization_endpoint')}?grant_type={config.get('grant_type')}&account_id={config.get('account_id')}"
        with requests_mock.Mocker() as m:
            m.post(url, json=except_token_response, request_headers=headers, status_code=HTTPStatus.OK)
            self.assertEqual(client.generate_access_token(), except_access_token)
        with requests_mock.Mocker() as m:
            m.post(url, exc=requests.exceptions.RequestException)
            with self.assertRaises(Exception) as cm:
                client.generate_access_token()
            self.assertIn('Error while generating access token', str(cm.exception))
if __name__ == '__main__':
    unittest.main()