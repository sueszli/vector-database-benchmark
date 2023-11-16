import pydocumentdb.errors as errors

def find_collection(client, dbid, id):
    if False:
        for i in range(10):
            print('nop')
    'Find whether or not a CosmosDB collection exists.\n\n    Args:\n        client (object): A pydocumentdb client object.\n        dbid (str): Database ID.\n        id (str): Collection ID.\n\n    Returns:\n        bool: True if the collection exists, False otherwise.\n    '
    database_link = 'dbs/' + dbid
    collections = list(client.QueryCollections(database_link, {'query': 'SELECT * FROM r WHERE r.id=@id', 'parameters': [{'name': '@id', 'value': id}]}))
    if len(collections) > 0:
        return True
    else:
        return False

def read_collection(client, dbid, id):
    if False:
        print('Hello World!')
    'Read a CosmosDB collection.\n\n    Args:\n        client (object): A pydocumentdb client object.\n        dbid (str): Database ID.\n        id (str): Collection ID.\n\n    Returns:\n        object: A collection.\n    '
    try:
        database_link = 'dbs/' + dbid
        collection_link = database_link + '/colls/{0}'.format(id)
        collection = client.ReadCollection(collection_link)
        return collection
    except errors.DocumentDBError as e:
        if e.status_code == 404:
            print("A collection with id '{0}' does not exist".format(id))
        else:
            raise errors.HTTPFailure(e.status_code)

def read_database(client, id):
    if False:
        return 10
    'Read a CosmosDB database.\n\n    Args:\n        client (object): A pydocumentdb client object.\n        id (str): Database ID.\n\n    Returns:\n        object: A database.\n    '
    try:
        database_link = 'dbs/' + id
        database = client.ReadDatabase(database_link)
        return database
    except errors.DocumentDBError as e:
        if e.status_code == 404:
            print("A database with id '{0}' does not exist".format(id))
        else:
            raise errors.HTTPFailure(e.status_code)

def find_database(client, id):
    if False:
        print('Hello World!')
    'Find whether or not a CosmosDB database exists.\n\n    Args:\n        client (object): A pydocumentdb client object.\n        id (str): Database ID.\n\n    Returns:\n        bool: True if the database exists, False otherwise.\n    '
    databases = list(client.QueryDatabases({'query': 'SELECT * FROM r WHERE r.id=@id', 'parameters': [{'name': '@id', 'value': id}]}))
    if len(databases) > 0:
        return True
    else:
        return False