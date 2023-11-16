import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos import ThroughputProperties
import config
HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']

def find_database(client, id):
    if False:
        i = 10
        return i + 15
    print('1. Query for Database')
    databases = list(client.query_databases({'query': 'SELECT * FROM r WHERE r.id=@id', 'parameters': [{'name': '@id', 'value': id}]}))
    if len(databases) > 0:
        print("Database with id '{0}' was found".format(id))
    else:
        print("No database with id '{0}' was found".format(id))

def create_database(client, id):
    if False:
        print('Hello World!')
    print('\n2. Create Database')
    try:
        client.create_database(id=id)
        print("Database with id '{0}' created".format(id))
    except exceptions.CosmosResourceExistsError:
        print("A database with id '{0}' already exists".format(id))
    print('\n2.8 Create Database - With auto scale settings')
    try:
        client.create_database(id=id, offer_throughput=ThroughputProperties(auto_scale_max_throughput=5000, auto_scale_increment_percent=0))
        print("Database with id '{0}' created".format(id))
    except exceptions.CosmosResourceExistsError:
        print("A database with id '{0}' already exists".format(id))

def read_database(client, id):
    if False:
        while True:
            i = 10
    print('\n3. Get a Database by id')
    try:
        database = client.get_database_client(id)
        database.read()
        print("Database with id '{0}' was found, it's link is {1}".format(id, database.database_link))
    except exceptions.CosmosResourceNotFoundError:
        print("A database with id '{0}' does not exist".format(id))

def list_databases(client):
    if False:
        return 10
    print('\n4. List all Databases on an account')
    print('Databases:')
    databases = list(client.list_databases())
    if not databases:
        return
    for database in databases:
        print(database['id'])

def delete_database(client, id):
    if False:
        for i in range(10):
            print('nop')
    print('\n5. Delete Database')
    try:
        client.delete_database(id)
        print("Database with id '{0}' was deleted".format(id))
    except exceptions.CosmosResourceNotFoundError:
        print("A database with id '{0}' does not exist".format(id))

def run_sample():
    if False:
        i = 10
        return i + 15
    client = cosmos_client.CosmosClient(HOST, {'masterKey': MASTER_KEY})
    try:
        find_database(client, DATABASE_ID)
        create_database(client, DATABASE_ID)
        read_database(client, DATABASE_ID)
        list_databases(client)
        delete_database(client, DATABASE_ID)
    except exceptions.CosmosHttpResponseError as e:
        print('\nrun_sample has caught an error. {0}'.format(e.message))
    finally:
        print('\nrun_sample done')
if __name__ == '__main__':
    run_sample()