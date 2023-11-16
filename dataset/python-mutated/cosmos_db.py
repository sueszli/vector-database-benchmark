import os
import uuid
from azure.cosmos import CosmosClient
from azure.cosmos.partition_key import PartitionKey

class CosmosDB:

    def __init__(self):
        if False:
            while True:
                i = 10
        URL = os.environ['COSMOS_ENDPOINT']
        KEY = os.environ['COSMOS_KEY']
        self.client = CosmosClient(URL, {'masterKey': KEY})
        self.dbName = 'pySolarSystem-' + uuid.uuid1().hex

    def create_database(self):
        if False:
            while True:
                i = 10
        print("Creating '{0}' database...".format(self.dbName))
        return self.client.create_database(self.dbName)

    def create_container(self, db):
        if False:
            while True:
                i = 10
        collectionName = 'Planets'
        print("Creating '{0}' collection...".format(collectionName))
        partition_key = PartitionKey(path='/id', kind='Hash')
        return db.create_container(id='Planets', partition_key=partition_key)

    def create_documents(self, container):
        if False:
            return 10
        planets = [{'id': 'Earth', 'HasRings': False, 'Radius': 3959, 'Moons': [{'Name': 'Moon'}]}, {'id': 'Mars', 'HasRings': False, 'Radius': 2106, 'Moons': [{'Name': 'Phobos'}, {'Name': 'Deimos'}]}]
        print('Inserting items in the collection...')
        for planet in planets:
            container.create_item(planet)
            print("\t'{0}' created".format(planet['id']))
        print('\tdone')

    def simple_query(self, container):
        if False:
            for i in range(10):
                print('nop')
        print('Quering the container...')
        items = list(container.query_items(query='SELECT c.id FROM c', enable_cross_partition_query=True))
        print('\tdone: {0}'.format(items))

    def delete_database(self):
        if False:
            while True:
                i = 10
        print('Cleaning up the resource...')
        self.client.delete_database(self.dbName)
        print('\tdone')

    def run(self):
        if False:
            return 10
        print('')
        print('------------------------')
        print('Cosmos DB')
        print('------------------------')
        print('1) Create a Database')
        print('2) Create a Container in the database')
        print('3) Insert Documents (items) into the Container')
        print('4) Delete Database (Clean up the resource)')
        print('')
        try:
            self.delete_database()
        except:
            pass
        try:
            db = self.create_database()
            container = self.create_container(db=db)
            self.create_documents(container=container)
            self.simple_query(container=container)
        finally:
            self.delete_database()