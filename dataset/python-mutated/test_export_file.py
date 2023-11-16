import graphene
from .....core import JobStatus
from .....csv import ExportEvents
from ....tests.utils import get_graphql_content, get_graphql_content_from_response
EXPORT_FILE_QUERY = '\n    query($id: ID!){\n        exportFile(id: $id){\n            id\n            status\n            createdAt\n            updatedAt\n            url\n            user{\n                email\n            }\n            app{\n                name\n            }\n            events{\n                date\n                type\n                user{\n                    email\n                }\n                message\n                app{\n                    name\n                }\n            }\n        }\n    }\n'
EXPORT_FILE_BY_APP_QUERY = '\n    query($id: ID!){\n        exportFile(id: $id){\n            id\n            status\n            createdAt\n            updatedAt\n            url\n            app{\n                name\n            }\n            events{\n                date\n                type\n                message\n                app{\n                    name\n                }\n            }\n        }\n    }\n'

def test_query_export_file(staff_api_client, user_export_file, permission_manage_products, permission_manage_apps, user_export_event):
    if False:
        i = 10
        return i + 15
    query = EXPORT_FILE_QUERY
    variables = {'id': graphene.Node.to_global_id('ExportFile', user_export_file.pk)}
    response = staff_api_client.post_graphql(query, variables=variables, permissions=[permission_manage_products, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['exportFile']
    assert data['status'] == JobStatus.PENDING.upper()
    assert data['createdAt']
    assert data['updatedAt']
    assert data['app'] is None
    assert not data['url']
    assert data['user']['email'] == staff_api_client.user.email
    assert len(data['events']) == 1
    event = data['events'][0]
    assert event['date']
    assert event['message'] == user_export_event.parameters.get('message')
    assert event['type'] == ExportEvents.EXPORT_FAILED.upper()
    assert event['user']['email'] == user_export_event.user.email
    assert event['app'] is None

def test_query_export_file_by_app(app_api_client, user_export_file, permission_manage_products, permission_manage_apps, user_export_event):
    if False:
        while True:
            i = 10
    query = EXPORT_FILE_BY_APP_QUERY
    variables = {'id': graphene.Node.to_global_id('ExportFile', user_export_file.pk)}
    response = app_api_client.post_graphql(query, variables=variables, permissions=[permission_manage_products, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['exportFile']
    assert data['status'] == JobStatus.PENDING.upper()
    assert data['createdAt']
    assert data['updatedAt']
    assert data['app'] is None
    assert not data['url']
    assert len(data['events']) == 1
    event = data['events'][0]
    assert event['date']
    assert event['message'] == user_export_event.parameters.get('message')
    assert event['type'] == ExportEvents.EXPORT_FAILED.upper()
    assert event['app'] is None

def test_query_export_file_export_file_with_app(app, staff_api_client, app_export_file, permission_manage_products, permission_manage_apps, permission_manage_staff, app_export_event):
    if False:
        i = 10
        return i + 15
    query = EXPORT_FILE_BY_APP_QUERY
    variables = {'id': graphene.Node.to_global_id('ExportFile', app_export_file.pk)}
    response = staff_api_client.post_graphql(query, variables=variables, permissions=[permission_manage_products, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['exportFile']
    assert data['status'] == JobStatus.PENDING.upper()
    assert data['createdAt']
    assert data['updatedAt']
    assert data['app']['name'] == app.name
    assert not data['url']
    assert len(data['events']) == 1
    event = data['events'][0]
    assert event['date']
    assert event['message'] == app_export_event.parameters.get('message')
    assert event['type'] == ExportEvents.EXPORT_FAILED.upper()
    assert event['app']['name'] == app.name

def test_query_export_file_as_app(app_api_client, user_export_file, permission_manage_products, permission_manage_staff, permission_manage_apps, user_export_event):
    if False:
        for i in range(10):
            print('nop')
    query = EXPORT_FILE_BY_APP_QUERY
    variables = {'id': graphene.Node.to_global_id('ExportFile', user_export_file.pk)}
    response = app_api_client.post_graphql(query, variables=variables, permissions=[permission_manage_products, permission_manage_apps])
    content = get_graphql_content(response)
    data = content['data']['exportFile']
    assert data['status'] == JobStatus.PENDING.upper()
    assert data['createdAt']
    assert data['updatedAt']
    assert data['app'] is None
    assert not data['url']
    assert len(data['events']) == 1
    event = data['events'][0]
    assert event['date']
    assert event['message'] == user_export_event.parameters.get('message')
    assert event['type'] == ExportEvents.EXPORT_FAILED.upper()
    assert event['app'] is None

def test_query_export_file_by_invalid_id(staff_api_client, user_export_file, permission_manage_products):
    if False:
        while True:
            i = 10
    id = 'bh/'
    variables = {'id': id}
    response = staff_api_client.post_graphql(EXPORT_FILE_QUERY, variables, permissions=[permission_manage_products])
    content = get_graphql_content_from_response(response)
    assert len(content['errors']) == 1
    assert content['errors'][0]['message'] == f'Invalid ID: {id}. Expected: ExportFile.'
    assert content['data']['exportFile'] is None

def test_query_export_file_with_invalid_object_type(staff_api_client, user_export_file, permission_manage_products):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('Order', user_export_file.pk)}
    response = staff_api_client.post_graphql(EXPORT_FILE_QUERY, variables, permissions=[permission_manage_products])
    content = get_graphql_content(response)
    assert content['data']['exportFile'] is None