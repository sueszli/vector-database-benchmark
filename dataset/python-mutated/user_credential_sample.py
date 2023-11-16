"""
FILE: user_credential_sample.py
DESCRIPTION:
    These samples demonstrate creating a `CommunicationTokenCredential` object.
    The `CommunicationTokenCredential` object is used to authenticate a user with Communication Services,
    such as Chat or Calling. It optionally provides an auto-refresh mechanism to ensure a continuously
    stable authentication state during communications.

USAGE:
    python user_credential_sample.py
    Set the environment variables with your own values before running the sample:
    1) COMMUNICATION_SAMPLES_CONNECTION_STRING - the connection string in your Communication Services resource
"""
import os
from azure.communication.chat import CommunicationTokenCredential
from azure.communication.identity import CommunicationIdentityClient

class CommunicationTokenCredentialSamples(object):
    connection_string = os.environ.get('COMMUNICATION_SAMPLES_CONNECTION_STRING', None)
    if not connection_string:
        raise ValueError('Set COMMUNICATION_SAMPLES_CONNECTION_STRING env before running this sample.')
    identity_client = CommunicationIdentityClient.from_connection_string(connection_string)
    user = identity_client.create_user()
    token_response = identity_client.get_token(user, scopes=['chat'])
    token = token_response.token

    def create_credential_with_static_token(self):
        if False:
            for i in range(10):
                print('nop')
        with CommunicationTokenCredential(self.token) as credential:
            token_response = credential.get_token()
            print('Token issued with value: ' + token_response.token)

    def create_credential_with_refreshing_callback(self):
        if False:
            return 10
        fetch_token_from_server = lambda : None
        with CommunicationTokenCredential(self.token, token_refresher=fetch_token_from_server) as credential:
            token_response = credential.get_token()
            print('Token issued with value: ' + token_response.token)

    def create_credential_with_proactive_refreshing_callback(self):
        if False:
            for i in range(10):
                print('nop')
        fetch_token_from_server = lambda : None
        with CommunicationTokenCredential(self.token, token_refresher=fetch_token_from_server, proactive_refresh=True) as credential:
            token_response = credential.get_token()
            print('Token issued with value: ' + token_response.token)

    def clean_up(self):
        if False:
            while True:
                i = 10
        print('cleaning up: deleting created user.')
        self.identity_client.delete_user(self.user)
if __name__ == '__main__':
    sample = CommunicationTokenCredentialSamples()
    sample.create_credential_with_static_token()
    sample.create_credential_with_refreshing_callback()
    sample.create_credential_with_proactive_refreshing_callback()
    sample.clean_up()