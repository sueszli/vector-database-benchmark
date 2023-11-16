from google.cloud.datastore_admin_v1 import DatastoreAdminClient

def client_create():
    if False:
        while True:
            i = 10
    'Creates a new Datastore admin client.'
    client = DatastoreAdminClient()
    print('Admin client created\n')
    return client

def export_entities(project_id, output_url_prefix):
    if False:
        print('Hello World!')
    '\n    Exports a copy of all or a subset of entities from\n    Datastore to another storage system, such as Cloud Storage.\n    '
    client = DatastoreAdminClient()
    op = client.export_entities({'project_id': project_id, 'output_url_prefix': output_url_prefix})
    response = op.result(timeout=300)
    print('Entities were exported\n')
    return response

def import_entities(project_id, input_url):
    if False:
        for i in range(10):
            print('nop')
    'Imports entities into Datastore.'
    client = DatastoreAdminClient()
    op = client.import_entities({'project_id': project_id, 'input_url': input_url})
    response = op.result(timeout=300)
    print('Entities were imported\n')
    return response

def get_index(project_id, index_id):
    if False:
        for i in range(10):
            print('nop')
    'Gets an index.'
    client = DatastoreAdminClient()
    index = client.get_index({'project_id': project_id, 'index_id': index_id})
    print('Got index: %v\n', index.index_id)
    return index

def list_indexes(project_id):
    if False:
        print('Hello World!')
    'Lists the indexes.'
    client = DatastoreAdminClient()
    indexes = []
    for index in client.list_indexes({'project_id': project_id}):
        indexes.append(index)
        print('Got index: %v\n', index.index_id)
    print('Got list of indexes\n')
    return indexes