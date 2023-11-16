import unittest
import pytest
import uuid
import azure.cosmos.partition_key as partition_key
import azure.cosmos.cosmos_client as cosmos_client
import test_config
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class PartitionKeyTests(unittest.TestCase):
    """Tests to verify if non partitioned collections are properly accessed on migration with version 2018-12-31.
    """
    host = test_config._test_config.host
    masterKey = test_config._test_config.masterKey
    connectionPolicy = test_config._test_config.connectionPolicy

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        cls.client.delete_database(test_config._test_config.TEST_DATABASE_ID)

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls.client = cosmos_client.CosmosClient(cls.host, cls.masterKey, consistency_level='Session', connection_policy=cls.connectionPolicy)
        cls.created_db = cls.client.create_database_if_not_exists(test_config._test_config.TEST_DATABASE_ID)
        cls.created_collection = cls.created_db.create_container_if_not_exists(id=test_config._test_config.TEST_COLLECTION_MULTI_PARTITION_WITH_CUSTOM_PK_ID, partition_key=partition_key.PartitionKey(path='/pk'))

    def test_multi_partition_collection_read_document_with_no_pk(self):
        if False:
            i = 10
            return i + 15
        document_definition = {'id': str(uuid.uuid4())}
        self.created_collection.create_item(body=document_definition)
        read_item = self.created_collection.read_item(item=document_definition['id'], partition_key=partition_key.NonePartitionKeyValue)
        self.assertEqual(read_item['id'], document_definition['id'])
        self.created_collection.delete_item(item=document_definition['id'], partition_key=partition_key.NonePartitionKeyValue)

    def test_hash_v2_partition_key_definition(self):
        if False:
            for i in range(10):
                print('nop')
        created_container = self.created_db.create_container(id='container_with_pkd_v2' + str(uuid.uuid4()), partition_key=partition_key.PartitionKey(path='/id', kind='Hash'))
        created_container_properties = created_container.read()
        self.assertEqual(created_container_properties['partitionKey']['version'], 2)
        self.created_db.delete_container(created_container)
        created_container = self.created_db.create_container(id='container_with_pkd_v2' + str(uuid.uuid4()), partition_key=partition_key.PartitionKey(path='/id', kind='Hash', version=2))
        created_container_properties = created_container.read()
        self.assertEqual(created_container_properties['partitionKey']['version'], 2)
        self.created_db.delete_container(created_container)

    def test_hash_v1_partition_key_definition(self):
        if False:
            print('Hello World!')
        created_container = self.created_db.create_container(id='container_with_pkd_v2' + str(uuid.uuid4()), partition_key=partition_key.PartitionKey(path='/id', kind='Hash', version=1))
        created_container_properties = created_container.read()
        self.assertEqual(created_container_properties['partitionKey']['version'], 1)
        self.created_db.delete_container(created_container)