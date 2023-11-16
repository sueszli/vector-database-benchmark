import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
from azure.cosmos import ThroughputProperties
import config
HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']
CONTAINER_ID = config.settings['container_id']

def find_container(db, id):
    if False:
        print('Hello World!')
    print('1. Query for Container')
    containers = list(db.query_containers({'query': 'SELECT * FROM r WHERE r.id=@id', 'parameters': [{'name': '@id', 'value': id}]}))
    if len(containers) > 0:
        print("Container with id '{0}' was found".format(id))
    else:
        print("No container with id '{0}' was found".format(id))

def create_container(db, id):
    if False:
        while True:
            i = 10
    ' Execute basic container creation.\n    This will create containers with 400 RUs with different indexing, partitioning, and storage options '
    partition_key = PartitionKey(path='/id', kind='Hash')
    print('\n2.1 Create Container - Basic')
    try:
        db.create_container(id=id, partition_key=partition_key)
        print("Container with id '{0}' created".format(id))
    except exceptions.CosmosResourceExistsError:
        print("A container with id '{0}' already exists".format(id))
    print('\n2.2 Create Container - With custom index policy')
    try:
        coll = {'id': id + '_container_custom_index_policy', 'indexingPolicy': {'automatic': False}}
        container = db.create_container(id=coll['id'], partition_key=partition_key, indexing_policy=coll['indexingPolicy'])
        properties = container.read()
        print("Container with id '{0}' created".format(container.id))
        print("IndexPolicy Mode - '{0}'".format(properties['indexingPolicy']['indexingMode']))
        print("IndexPolicy Automatic - '{0}'".format(properties['indexingPolicy']['automatic']))
    except exceptions.CosmosResourceExistsError:
        print("A container with id '{0}' already exists".format(coll['id']))
    print('\n2.3 Create Container - With custom provisioned throughput')
    try:
        container = db.create_container(id=id + '_container_custom_throughput', partition_key=partition_key, offer_throughput=400)
        print("Container with id '{0}' created".format(container.id))
    except exceptions.CosmosResourceExistsError:
        print("A container with id '{0}' already exists".format(coll['id']))
    print('\n2.4 Create Container - With Unique keys')
    try:
        container = db.create_container(id=id + '_container_unique_keys', partition_key=partition_key, unique_key_policy={'uniqueKeys': [{'paths': ['/field1/field2', '/field3']}]})
        properties = container.read()
        unique_key_paths = properties['uniqueKeyPolicy']['uniqueKeys'][0]['paths']
        print("Container with id '{0}' created".format(container.id))
        print("Unique Key Paths - '{0}', '{1}'".format(unique_key_paths[0], unique_key_paths[1]))
    except exceptions.CosmosResourceExistsError:
        print("A container with id 'container_unique_keys' already exists")
    print('\n2.5 Create Container - With Partition key V2 (Default)')
    try:
        container = db.create_container(id=id + '_container_partition_key_v2', partition_key=PartitionKey(path='/id', kind='Hash'))
        properties = container.read()
        print("Container with id '{0}' created".format(container.id))
        print("Partition Key - '{0}'".format(properties['partitionKey']))
    except exceptions.CosmosResourceExistsError:
        print("A container with id 'container_partition_key_v2' already exists")
    print('\n2.6 Create Container - With Partition key V1')
    try:
        container = db.create_container(id=id + '_container_partition_key_v1', partition_key=PartitionKey(path='/id', kind='Hash', version=1))
        properties = container.read()
        print("Container with id '{0}' created".format(container.id))
        print("Partition Key - '{0}'".format(properties['partitionKey']))
    except exceptions.CosmosResourceExistsError:
        print("A container with id 'container_partition_key_v1' already exists")
    print('\n2.7 Create Container - With analytical store enabled')
    try:
        container = db.create_container(id=id + '_container_analytical_store', partition_key=PartitionKey(path='/id', kind='Hash'), analytical_storage_ttl=None)
        'A value of None leaves analytical storage off and a value of -1 turns analytical storage on with no TTL.\n        Please note that analytical storage can only be enabled on Synapse Link enabled accounts.'
        properties = container.read()
        print("Container with id '{0}' created".format(container.id))
        print("Partition Key - '{0}'".format(properties['partitionKey']))
    except exceptions.CosmosResourceExistsError:
        print("A container with id '_container_analytical_store' already exists")
    print('\n2.8 Create Container - With auto scale settings')
    try:
        container = db.create_container(id=id + '_container_auto_scale_settings', partition_key=partition_key, offer_throughput=ThroughputProperties(auto_scale_max_throughput=5000, auto_scale_increment_percent=0))
        print("Container with id '{0}' created".format(container.id))
    except exceptions.CosmosResourceExistsError:
        print("A container with id '{0}' already exists".format(coll['id']))

def manage_provisioned_throughput(db, id):
    if False:
        i = 10
        return i + 15
    print('\n3.1 Get Container provisioned throughput (RU/s)')
    try:
        container = db.get_container_client(container=id)
        offer = container.get_throughput()
        print("Found Offer '{0}' for Container '{1}' and its throughput is '{2}'".format(offer.properties['id'], container.id, offer.properties['content']['offerThroughput']))
    except exceptions.CosmosResourceExistsError:
        print("A container with id '{0}' does not exist".format(id))
    print('\n3.2 Change Provisioned Throughput of Container')
    offer = container.replace_throughput(offer.offer_throughput + 100)
    print("Replaced Offer. Provisioned Throughput is now '{0}'".format(offer.properties['content']['offerThroughput']))

def read_Container(db, id):
    if False:
        print('Hello World!')
    print('\n4. Get a Container by id')
    try:
        container = db.get_container_client(id)
        container.read()
        print("Container with id '{0}' was found, it's link is {1}".format(container.id, container.container_link))
    except exceptions.CosmosResourceNotFoundError:
        print("A container with id '{0}' does not exist".format(id))

def list_Containers(db):
    if False:
        while True:
            i = 10
    print('\n5. List all Container in a Database')
    print('Containers:')
    containers = list(db.list_containers())
    if not containers:
        return
    for container in containers:
        print(container['id'])

def delete_Container(db, id):
    if False:
        while True:
            i = 10
    print('\n6. Delete Container')
    try:
        db.delete_container(id)
        print("Container with id '{0}' was deleted".format(id))
    except exceptions.CosmosResourceNotFoundError:
        print("A container with id '{0}' does not exist".format(id))

def run_sample():
    if False:
        return 10
    client = cosmos_client.CosmosClient(HOST, {'masterKey': MASTER_KEY})
    try:
        try:
            db = client.create_database(id=DATABASE_ID)
        except exceptions.CosmosResourceExistsError:
            db = client.get_database_client(DATABASE_ID)
        find_container(db, CONTAINER_ID)
        create_container(db, CONTAINER_ID)
        manage_provisioned_throughput(db, CONTAINER_ID)
        read_Container(db, CONTAINER_ID)
        list_Containers(db)
        delete_Container(db, CONTAINER_ID)
        try:
            client.delete_database(db)
        except exceptions.CosmosResourceNotFoundError:
            pass
    except exceptions.CosmosHttpResponseError as e:
        print('\nrun_sample has caught an error. {0}'.format(e.message))
    finally:
        print('\nrun_sample done')
if __name__ == '__main__':
    run_sample()