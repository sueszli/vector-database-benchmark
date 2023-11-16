import unittest
import uuid
import pytest
import azure.cosmos.documents as documents
import azure.cosmos.cosmos_client as cosmos_client
from azure.cosmos import PartitionKey
from azure.cosmos._routing.routing_map_provider import PartitionKeyRangeCache
from azure.cosmos._routing import routing_range as routing_range
import test_config
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class RoutingMapEndToEndTests(unittest.TestCase):
    """Routing Map Functionalities end to end Tests.
    """
    host = test_config._test_config.host
    masterKey = test_config._test_config.masterKey
    connectionPolicy = test_config._test_config.connectionPolicy

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        if cls.masterKey == '[YOUR_KEY_HERE]' or cls.host == '[YOUR_ENDPOINT_HERE]':
            raise Exception("You must specify your Azure Cosmos account values for 'masterKey' and 'host' at the top of this class to run the tests.")
        cls.client = cosmos_client.CosmosClient(cls.host, cls.masterKey, consistency_level='Session', connection_policy=cls.connectionPolicy)
        cls.created_database = cls.client.create_database_if_not_exists(test_config._test_config.TEST_DATABASE_ID)
        cls.created_container = cls.created_database.create_container('routing_map_tests_' + str(uuid.uuid4()), PartitionKey(path='/pk'))
        cls.collection_link = cls.created_container.container_link

    def test_read_partition_key_ranges(self):
        if False:
            i = 10
            return i + 15
        partition_key_ranges = list(self.client.client_connection._ReadPartitionKeyRanges(self.collection_link))
        if self.host == 'https://localhost:8081/':
            self.assertEqual(5, len(partition_key_ranges))
        else:
            self.assertEqual(1, len(partition_key_ranges))

    def test_routing_map_provider(self):
        if False:
            while True:
                i = 10
        partition_key_ranges = list(self.client.client_connection._ReadPartitionKeyRanges(self.collection_link))
        routing_mp = PartitionKeyRangeCache(self.client.client_connection)
        overlapping_partition_key_ranges = routing_mp.get_overlapping_ranges(self.collection_link, routing_range.Range('', 'FF', True, False))
        self.assertEqual(len(overlapping_partition_key_ranges), len(partition_key_ranges))
        self.assertEqual(overlapping_partition_key_ranges, partition_key_ranges)
if __name__ == '__main__':
    unittest.main()