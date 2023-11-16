import unittest
from unittest.mock import MagicMock
import pytest
import azure.cosmos.cosmos_client as cosmos_client
import test_config
from azure.cosmos.partition_key import PartitionKey
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class CorrelatedActivityIdTest(unittest.TestCase):
    configs = test_config._test_config
    host = configs.host
    masterKey = configs.masterKey

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.client = cosmos_client.CosmosClient(cls.host, cls.masterKey)
        cls.database = cls.client.create_database_if_not_exists(test_config._test_config.TEST_DATABASE_ID)
        cls.container = cls.database.create_container(id=test_config._test_config.TEST_COLLECTION_MULTI_PARTITION_ID, partition_key=PartitionKey(path='/id'))

    def side_effect_correlated_activity_id(self, *args):
        if False:
            print('Hello World!')
        assert args[2]['x-ms-cosmos-correlated-activityid']
        raise StopIteration

    def test_correlated_activity_id(self):
        if False:
            i = 10
            return i + 15
        query = 'SELECT * from c ORDER BY c._ts'
        cosmos_client_connection = self.container.client_connection
        cosmos_client_connection._CosmosClientConnection__Get = MagicMock(side_effect=self.side_effect_correlated_activity_id)
        try:
            self.container.query_items(query=query, partition_key='pk-1')
        except StopIteration:
            pass
if __name__ == '__main__':
    unittest.main()