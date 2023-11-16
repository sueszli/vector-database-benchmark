from azure.cosmos.aio import CosmosClient
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
from azure.identity.aio import ClientSecretCredential, DefaultAzureCredential
import config
import asyncio
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
        return 10
    test_item = {'id': 'Item_' + str(num), 'test_object': True, 'lastName': 'Smith'}
    return test_item

async def create_sample_resources():
    print('creating sample resources')
    async with CosmosClient(HOST, MASTER_KEY) as client:
        db = await client.create_database(DATABASE_ID)
        await db.create_container(id=CONTAINER_ID, partition_key=PARTITION_KEY)

async def delete_sample_resources():
    print('deleting sample resources')
    async with CosmosClient(HOST, MASTER_KEY) as client:
        await client.delete_database(DATABASE_ID)

async def run_sample():
    await create_sample_resources()
    async with ClientSecretCredential(tenant_id=TENANT_ID, client_id=CLIENT_ID, client_secret=CLIENT_SECRET) as aad_credentials:
        async with CosmosClient(HOST, aad_credentials) as aad_client:
            print('Showed ClientSecretCredential, now showing DefaultAzureCredential')
    async with DefaultAzureCredential() as aad_credentials:
        async with CosmosClient(HOST, aad_credentials) as aad_client:
            db = aad_client.get_database_client(DATABASE_ID)
            container = db.get_container_client(CONTAINER_ID)
            print('Container info: ' + str(container.read()))
            await container.create_item(get_test_item(879))
            print('Point read result: ' + str(container.read_item(item='Item_0', partition_key='Item_0')))
            query_results = [item async for item in container.query_items(query='select * from c', partition_key='Item_0')]
            assert len(query_results) == 1
            print('Query result: ' + str(query_results[0]))
            await container.delete_item(item='Item_0', partition_key='Item_0')
            try:
                await aad_client.delete_database(DATABASE_ID)
            except exceptions.CosmosHttpResponseError as e:
                assert e.status_code == 403
                print('403 error assertion success')
    await delete_sample_resources()
    print('end of sample')
if __name__ == '__main__':
    asyncio.run(run_sample())