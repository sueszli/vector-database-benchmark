from azure.cosmos import CosmosClient
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
from azure.identity import ClientSecretCredential, DefaultAzureCredential
import config
HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
TENANT_ID = config.settings['tenant_id']
CLIENT_ID = config.settings['client_id']
CLIENT_SECRET = config.settings['client_secret']
DATABASE_ID = config.settings['database_id']
CONTAINER_ID = config.settings['container_id']
PARTITION_KEY = PartitionKey(path='/id')

def get_test_item(num):
    if False:
        for i in range(10):
            print('nop')
    test_item = {'id': 'Item_' + str(num), 'test_object': True, 'lastName': 'Smith'}
    return test_item

def create_sample_resources():
    if False:
        i = 10
        return i + 15
    print('creating sample resources')
    client = CosmosClient(HOST, MASTER_KEY)
    db = client.create_database(DATABASE_ID)
    db.create_container(id=CONTAINER_ID, partition_key=PARTITION_KEY)

def delete_sample_resources():
    if False:
        while True:
            i = 10
    print('deleting sample resources')
    client = CosmosClient(HOST, MASTER_KEY)
    client.delete_database(DATABASE_ID)

def run_sample():
    if False:
        return 10
    create_sample_resources()
    aad_credentials = ClientSecretCredential(tenant_id=TENANT_ID, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    aad_credentials = DefaultAzureCredential()
    aad_client = CosmosClient(HOST, aad_credentials)
    db = aad_client.get_database_client(DATABASE_ID)
    container = db.get_container_client(CONTAINER_ID)
    print('Container info: ' + str(container.read()))
    container.create_item(get_test_item(0))
    print('Point read result: ' + str(container.read_item(item='Item_0', partition_key='Item_0')))
    query_results = list(container.query_items(query='select * from c', partition_key='Item_0'))
    assert len(query_results) == 1
    print('Query result: ' + str(query_results[0]))
    container.delete_item(item='Item_0', partition_key='Item_0')
    try:
        aad_client.delete_database(DATABASE_ID)
    except exceptions.CosmosHttpResponseError as e:
        assert e.status_code == 403
        print('403 error assertion success')
    delete_sample_resources()
    print('end of sample')
if __name__ == '__main__':
    run_sample()