import unittest
import azure.cosmos.cosmos_client as cosmos_client
from azure.cosmos import http_constants, exceptions, PartitionKey
import pytest
import uuid
from test_config import _test_config
pytestmark = pytest.mark.cosmosEmulator
DATABASE_ID = 'PythonSDKUserConfigTesters'
CONTAINER_ID = 'PythonSDKTestContainer'

def get_test_item():
    if False:
        while True:
            i = 10
    item = {'id': 'Async_' + str(uuid.uuid4()), 'test_object': True, 'lastName': 'Smith'}
    return item

@pytest.mark.usefixtures('teardown')
class TestUserConfigs(unittest.TestCase):

    def test_invalid_connection_retry_configuration(self):
        if False:
            print('Hello World!')
        try:
            cosmos_client.CosmosClient(url=_test_config.host, credential=_test_config.masterKey, consistency_level='Session', connection_retry_policy='Invalid Policy')
        except TypeError as e:
            self.assertTrue(str(e).startswith('Unsupported retry policy'))

    def test_enable_endpoint_discovery(self):
        if False:
            print('Hello World!')
        client_false = cosmos_client.CosmosClient(url=_test_config.host, credential=_test_config.masterKey, consistency_level='Session', enable_endpoint_discovery=False)
        client_default = cosmos_client.CosmosClient(url=_test_config.host, credential=_test_config.masterKey, consistency_level='Session')
        client_true = cosmos_client.CosmosClient(url=_test_config.host, credential=_test_config.masterKey, consistency_level='Session', enable_endpoint_discovery=True)
        self.assertFalse(client_false.client_connection.connection_policy.EnableEndpointDiscovery)
        self.assertTrue(client_default.client_connection.connection_policy.EnableEndpointDiscovery)
        self.assertTrue(client_true.client_connection.connection_policy.EnableEndpointDiscovery)

    def test_authentication_error(self):
        if False:
            while True:
                i = 10
        try:
            cosmos_client.CosmosClient(url=_test_config.host, credential='wrong_key')
        except exceptions.CosmosHttpResponseError as e:
            self.assertEqual(e.status_code, http_constants.StatusCodes.UNAUTHORIZED)

    def test_default_account_consistency(self):
        if False:
            i = 10
            return i + 15
        if _test_config.host != 'https://localhost:8081/':
            return
        client = cosmos_client.CosmosClient(url=_test_config.host, credential=_test_config.masterKey)
        database_account = client.get_database_account()
        account_consistency_level = database_account.ConsistencyPolicy['defaultConsistencyLevel']
        self.assertEqual(account_consistency_level, 'Session')
        database = client.create_database(DATABASE_ID)
        container = database.create_container(id=CONTAINER_ID, partition_key=PartitionKey(path='/id'))
        container.create_item(body=get_test_item())
        session_token = client.client_connection.last_response_headers[http_constants.CookieHeaders.SessionToken]
        item2 = get_test_item()
        container.create_item(body=item2)
        session_token2 = client.client_connection.last_response_headers[http_constants.CookieHeaders.SessionToken]
        self.assertNotEqual(session_token, session_token2)
        container.read_item(item=item2.get('id'), partition_key=item2.get('id'))
        read_session_token = client.client_connection.last_response_headers[http_constants.CookieHeaders.SessionToken]
        self.assertEqual(session_token2, read_session_token)
        client.delete_database(DATABASE_ID)
        custom_level = 'Eventual'
        client = cosmos_client.CosmosClient(url=_test_config.host, credential=_test_config.masterKey, consistency_level=custom_level)
        database_account = client.get_database_account()
        account_consistency_level = database_account.ConsistencyPolicy['defaultConsistencyLevel']
        self.assertNotEqual(client.client_connection.default_headers[http_constants.HttpHeaders.ConsistencyLevel], account_consistency_level)
        custom_level = 'Strong'
        client = cosmos_client.CosmosClient(url=_test_config.host, credential=_test_config.masterKey, consistency_level=custom_level)
        try:
            client.create_database(DATABASE_ID)
        except exceptions.CosmosHttpResponseError as e:
            self.assertEqual(e.status_code, http_constants.StatusCodes.BAD_REQUEST)
if __name__ == '__main__':
    unittest.main()