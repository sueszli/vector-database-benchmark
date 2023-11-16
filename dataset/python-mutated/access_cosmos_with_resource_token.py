import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import azure.cosmos.documents as documents
import config
import json
HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']
CONTAINER_ID = config.settings['container_id']
PARTITION_KEY = PartitionKey(path='/username')
(USERNAME, USERNAME_2) = ('user', 'user2')
CONTAINER_ALL_PERMISSION = 'CONTAINER_ALL_PERMISSION'
PARTITION_READ_PERMISSION = 'PARTITION_READ_PERMISSION'
DOCUMENT_ALL_PERMISSION = 'DOCUMENT_ALL_PERMISSION'

def create_user_if_not_exists(db, username):
    if False:
        for i in range(10):
            print('nop')
    try:
        user = db.create_user(body={'id': username})
    except exceptions.CosmosResourceExistsError:
        user = db.get_user_client(username)
    return user

def create_permission_if_not_exists(user, permission_definition):
    if False:
        while True:
            i = 10
    try:
        permission = user.create_permission(permission_definition)
    except exceptions.CosmosResourceExistsError:
        permission = user.get_permission(permission_definition['id'])
    return permission

def token_client_upsert(container, username, item_id):
    if False:
        i = 10
        return i + 15
    try:
        container.upsert_item({'id': item_id, 'username': username, 'msg': 'This is a message for ' + username})
    except exceptions.CosmosHttpResponseError:
        print("Error in upserting item with id '{0}'.".format(item_id))

def token_client_read_all(container):
    if False:
        print('Hello World!')
    try:
        items = list(container.read_all_items())
        for i in items:
            print(i)
    except exceptions.CosmosResourceNotFoundError:
        print("Cannot read items--container '{0}' not found.".format(container.id))
    except exceptions.CosmosHttpResponseError:
        print("Error in reading items in container '{0}'.".format(container.id))

def token_client_read_item(container, username, item_id):
    if False:
        for i in range(10):
            print('nop')
    try:
        item = container.read_item(item=item_id, partition_key=username)
        print(item)
    except exceptions.CosmosResourceNotFoundError:
        print("Cannot read--item with id '{0}' not found.".format(item_id))
    except exceptions.CosmosHttpResponseError:
        print("Error in reading item with id '{0}'.".format(item_id))

def token_client_delete(container, username, item_id):
    if False:
        return 10
    try:
        container.delete_item(item=item_id, partition_key=username)
    except exceptions.CosmosResourceNotFoundError:
        print("Cannot delete--item with id '{0}' not found.".format(item_id))
    except exceptions.CosmosHttpResponseError:
        print("Error in deleting item with id '{0}'.".format(item_id))

def token_client_query(container, username):
    if False:
        return 10
    try:
        for item in container.query_items(query='SELECT * FROM my_container c WHERE c.username=@username', parameters=[{'name': '@username', 'value': username}], partition_key=username):
            print(json.dumps(item, indent=True))
    except exceptions.CosmosHttpResponseError:
        print('Error in querying item(s)')

def run_sample():
    if False:
        for i in range(10):
            print('nop')
    client = cosmos_client.CosmosClient(HOST, {'masterKey': MASTER_KEY})
    try:
        try:
            db = client.create_database(DATABASE_ID)
        except exceptions.CosmosResourceExistsError:
            db = client.get_database_client(DATABASE_ID)
        try:
            container = db.create_container(id=CONTAINER_ID, partition_key=PARTITION_KEY)
        except exceptions.CosmosResourceExistsError:
            container = db.get_container_client(CONTAINER_ID)
        user = create_user_if_not_exists(db, USERNAME)
        permission_definition = {'id': CONTAINER_ALL_PERMISSION, 'permissionMode': documents.PermissionMode.All, 'resource': container.container_link}
        permission = create_permission_if_not_exists(user, permission_definition)
        token = {}
        token[container.container_link] = permission.properties['_token']
        token_client = cosmos_client.CosmosClient(HOST, token)
        token_db = token_client.get_database_client(DATABASE_ID)
        token_container = token_db.get_container_client(CONTAINER_ID)
        (ITEM_1_ID, ITEM_2_ID, ITEM_3_ID) = ('1', '2', '3')
        token_client_upsert(token_container, USERNAME, ITEM_1_ID)
        token_client_upsert(token_container, USERNAME, ITEM_2_ID)
        token_client_upsert(token_container, USERNAME_2, ITEM_3_ID)
        token_client_read_all(token_container)
        token_client_read_item(token_container, USERNAME, ITEM_2_ID)
        token_client_query(token_container, USERNAME_2)
        token_client_delete(token_container, USERNAME, ITEM_2_ID)
        user_2 = create_user_if_not_exists(db, USERNAME_2)
        permission_definition = {'id': PARTITION_READ_PERMISSION, 'permissionMode': documents.PermissionMode.Read, 'resource': container.container_link, 'resourcePartitionKey': [USERNAME_2]}
        permission = create_permission_if_not_exists(user_2, permission_definition)
        read_token = {}
        read_token[container.container_link] = permission.properties['_token']
        token_client = cosmos_client.CosmosClient(HOST, read_token)
        token_db = token_client.get_database_client(DATABASE_ID)
        token_container = token_db.get_container_client(CONTAINER_ID)
        token_client_read_all(token_container)
        token_client_read_item(token_container, USERNAME_2, ITEM_3_ID)
        token_client_upsert(token_container, USERNAME_2, ITEM_3_ID)
        item_3 = token_container.read_item(item=ITEM_3_ID, partition_key=USERNAME_2)
        permission_list = list(user_2.list_permissions())
        for p in permission_list:
            user_2.delete_permission(p.get('id'))
        assert len(list(user_2.list_permissions())) == 0
        permission_definition = {'id': DOCUMENT_ALL_PERMISSION, 'permissionMode': documents.PermissionMode.All, 'resource': item_3.get('_self')}
        permission = create_permission_if_not_exists(user_2, permission_definition)
        item_token = {}
        item_token[container.container_link] = permission.properties['_token']
        token_client = cosmos_client.CosmosClient(HOST, item_token)
        token_db = token_client.get_database_client(DATABASE_ID)
        token_container = token_db.get_container_client(CONTAINER_ID)
        token_client_read_all(token_container)
        token_client_read_item(token_container, USERNAME, ITEM_1_ID)
        token_client_read_item(token_container, USERNAME_2, ITEM_3_ID)
        token_client_delete(token_container, USERNAME_2, ITEM_3_ID)
    except exceptions.CosmosHttpResponseError as e:
        print('\nrun_sample has caught an error. {0}'.format(e.message))
    finally:
        print('\nrun_sample done')
if __name__ == '__main__':
    run_sample()