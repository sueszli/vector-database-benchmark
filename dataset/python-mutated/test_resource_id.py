import uuid
import pytest
import test_config
from azure.cosmos import CosmosClient, PartitionKey
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class TestResourceIds:
    configs = test_config._test_config
    host = configs.host
    masterKey = configs.masterKey
    connectionPolicy = configs.connectionPolicy
    last_headers = []

    @classmethod
    def _set_up(cls):
        if False:
            return 10
        if cls.masterKey == '[YOUR_KEY_HERE]' or cls.host == '[YOUR_ENDPOINT_HERE]':
            raise Exception("You must specify your Azure Cosmos account values for 'masterKey' and 'host' at the top of this class to run the tests.")
        cls.client = CosmosClient(cls.host, cls.masterKey)
        cls.created_database = cls.client.create_database_if_not_exists(test_config._test_config.TEST_DATABASE_ID)

    def test_id_unicode_validation(self):
        if False:
            return 10
        self._set_up()
        resource_id1 = u'हिन्दी भारत की राष्ट्रीय भाषा है'
        resource_id2 = "!@$%^&*()-~`'_[]{}|;:,.<>"
        created_db1 = self.client.create_database_if_not_exists(resource_id1)
        created_db2 = self.client.create_database_if_not_exists(resource_id2)
        assert resource_id1 == created_db1.id
        assert resource_id2 == created_db2.id
        created_collection1 = created_db1.create_container_if_not_exists(id=resource_id1, partition_key=PartitionKey(path='/id', kind='Hash'))
        created_collection2 = created_db2.create_container_if_not_exists(id=resource_id2, partition_key=PartitionKey(path='/id', kind='Hash'))
        assert resource_id1 == created_collection1.id
        assert resource_id2 == created_collection2.id
        item1 = created_collection1.upsert_item({'id': resource_id1})
        item2 = created_collection1.upsert_item({'id': resource_id2})
        assert resource_id1 == item1.get('id')
        assert resource_id2 == item2.get('id')

    def test_create_illegal_characters_async(self):
        if False:
            for i in range(10):
                print('nop')
        self._set_up()
        database_id = str(uuid.uuid4())
        container_id = str(uuid.uuid4())
        partition_key = PartitionKey(path='/id')
        created_database = self.client.create_database(id=database_id)
        created_container = created_database.create_container(id=container_id, partition_key=partition_key)
        error_strings = ['Id contains illegal chars.', 'Id ends with a space.']
        illegal_strings = ['ID_with_back/slash', 'ID_\\with_forward_slashes', 'ID_with_question?_mark', 'ID_with_#pound', 'ID_with_tab\t', 'ID\r_with_return_carriage', 'ID_with_newline\n', 'ID_with_trailing_spaces   ']
        for resource_id in illegal_strings:
            try:
                self.client.create_database(resource_id)
                pytest.fail('Database create should have failed for id {}'.format(resource_id))
            except ValueError as e:
                assert str(e) in error_strings
            try:
                created_database.create_container(id=resource_id, partition_key=partition_key)
                pytest.fail('Container create should have failed for id {}'.format(resource_id))
            except ValueError as e:
                assert str(e) in error_strings
            try:
                created_container.create_item({'id': resource_id})
                pytest.fail('Item create should have failed for id {}'.format(resource_id))
            except ValueError as e:
                assert str(e) in error_strings
            try:
                created_container.upsert_item({'id': resource_id})
                pytest.fail('Item upsert should have failed for id {}'.format(resource_id))
            except ValueError as e:
                assert str(e) in error_strings