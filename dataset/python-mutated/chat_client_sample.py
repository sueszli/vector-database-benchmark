"""
FILE: chat_client_sample.py
DESCRIPTION:
    These samples demonstrate create a chat client, get a chat thread client,
    create a chat thread, get a chat thread by id, list chat threads, delete
    a chat thread by id.
    You need to use azure.communication.configuration module to get user access
    token and user identity before run this sample

USAGE:
    python chat_client_sample.py
    Set the environment variables with your own values before running the sample:
    1) AZURE_COMMUNICATION_SERVICE_ENDPOINT - Communication Service endpoint url
    2) TOKEN - the user access token, from token_response.token
    3) USER_ID - the user id, from token_response.identity
"""
import os

class ChatClientSamples(object):
    from azure.communication.identity import CommunicationIdentityClient
    connection_string = os.environ.get('COMMUNICATION_SAMPLES_CONNECTION_STRING', None)
    if not connection_string:
        raise ValueError('Set COMMUNICATION_SAMPLES_CONNECTION_STRING env before run this sample.')
    identity_client = CommunicationIdentityClient.from_connection_string(connection_string)
    user = identity_client.create_user()
    tokenresponse = identity_client.get_token(user, scopes=['chat'])
    token = tokenresponse.token
    endpoint = os.environ.get('AZURE_COMMUNICATION_SERVICE_ENDPOINT', None)
    if not endpoint:
        raise ValueError('Set AZURE_COMMUNICATION_SERVICE_ENDPOINT env before run this sample.')
    _thread_id = None

    def create_chat_client(self):
        if False:
            return 10
        token = self.token
        endpoint = self.endpoint
        from azure.communication.chat import ChatClient, CommunicationTokenCredential
        chat_client = ChatClient(endpoint, CommunicationTokenCredential(token))

    def create_thread(self):
        if False:
            print('Hello World!')
        token = self.token
        endpoint = self.endpoint
        user = self.user
        from datetime import datetime
        from azure.communication.chat import ChatClient, ChatParticipant, CommunicationUserIdentifier, CommunicationTokenCredential
        chat_client = ChatClient(endpoint, CommunicationTokenCredential(token))
        topic = 'test topic'
        participants = [ChatParticipant(identifier=user, display_name='name', share_history_time=datetime.utcnow())]
        create_chat_thread_result = chat_client.create_chat_thread(topic, thread_participants=participants)
        idempotency_token = 'b66d6031-fdcc-41df-8306-e524c9f226b8'
        create_chat_thread_result_w_repeatability_id = chat_client.create_chat_thread(topic, thread_participants=participants, idempotency_token=idempotency_token)
        self._thread_id = create_chat_thread_result.chat_thread.id
        print('thread created, id: ' + self._thread_id)

    def get_chat_thread_client(self):
        if False:
            while True:
                i = 10
        token = self.token
        endpoint = self.endpoint
        thread_id = self._thread_id
        from azure.communication.chat import ChatClient, CommunicationTokenCredential
        chat_client = ChatClient(endpoint, CommunicationTokenCredential(token))
        chat_thread_client = chat_client.get_chat_thread_client(thread_id)
        print('get_chat_thread_client succeeded with thread id: ', chat_thread_client.thread_id)

    def list_threads(self):
        if False:
            return 10
        token = self.token
        endpoint = self.endpoint
        from azure.communication.chat import ChatClient, CommunicationTokenCredential
        from datetime import datetime, timedelta
        chat_client = ChatClient(endpoint, CommunicationTokenCredential(token))
        start_time = datetime.utcnow() - timedelta(days=2)
        chat_threads = chat_client.list_chat_threads(results_per_page=5, start_time=start_time)
        print('list_threads succeeded with results_per_page is 5, and were created since 2 days ago.')
        for chat_thread_item_page in chat_threads.by_page():
            for chat_thread_item in chat_thread_item_page:
                print('thread id:', chat_thread_item.id)

    def delete_thread(self):
        if False:
            while True:
                i = 10
        token = self.token
        endpoint = self.endpoint
        thread_id = self._thread_id
        from azure.communication.chat import ChatClient, CommunicationTokenCredential
        chat_client = ChatClient(endpoint, CommunicationTokenCredential(token))
        chat_client.delete_chat_thread(thread_id)
        print('delete_thread succeeded')

    def clean_up(self):
        if False:
            for i in range(10):
                print('nop')
        print('cleaning up: deleting created user.')
        self.identity_client.delete_user(self.user)
if __name__ == '__main__':
    sample = ChatClientSamples()
    sample.create_chat_client()
    sample.create_thread()
    sample.get_chat_thread_client()
    sample.list_threads()
    sample.delete_thread()
    sample.clean_up()