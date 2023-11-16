import graphene
import pytest
from freezegun import freeze_time
from .....app.models import App
from .....app.types import AppType
from .....core.jwt import create_access_token_for_app
from .....webhook.models import Webhook
from ....tests.utils import assert_no_permission, get_graphql_content
from ...enums import AppTypeEnum
QUERY_APPS_WITH_FILTER = '\n    query ($filter: AppFilterInput ){\n        apps(first: 5, filter: $filter){\n            edges{\n                node{\n                    id\n                    isActive\n                    permissions{\n                        name\n                        code\n                    }\n                    tokens{\n                        authToken\n                    }\n                    webhooks{\n                        name\n                    }\n                    name\n                    type\n                    aboutApp\n                    dataPrivacy\n                    dataPrivacyUrl\n                    homepageUrl\n                    supportUrl\n                    configurationUrl\n                    appUrl\n                    metafield(key: "test")\n                    metafields(keys: ["test"])\n                    extensions{\n                        id\n                        label\n                        url\n                        mount\n                        target\n                        permissions{\n                            code\n                        }\n                    }\n                }\n            }\n        }\n    }\n'

@pytest.mark.parametrize(('app_filter', 'count'), [({'search': 'Sample'}, 1), ({'isActive': False}, 1), ({}, 2), ({'type': AppTypeEnum.THIRDPARTY.name}, 1), ({'type': AppTypeEnum.LOCAL.name}, 1)])
def test_apps_query(staff_api_client, permission_manage_apps, permission_manage_orders, app_with_extensions, external_app, app_filter, count):
    if False:
        i = 10
        return i + 15
    (app, app_extensions) = app_with_extensions
    external_app.is_active = False
    external_app.save()
    webhooks = Webhook.objects.bulk_create([Webhook(app=app, name='first', target_url='http://www.example.com/test'), Webhook(app=external_app, name='second', target_url='http://www.exa.com/s')])
    webhooks_names = [w.name for w in webhooks]
    variables = {'filter': app_filter}
    response = staff_api_client.post_graphql(QUERY_APPS_WITH_FILTER, variables, permissions=[permission_manage_apps, permission_manage_orders])
    content = get_graphql_content(response)
    apps_data = content['data']['apps']['edges']
    for app_data in apps_data:
        tokens = app_data['node']['tokens']
        assert len(tokens) == 1
        assert len(tokens[0]['authToken']) == 4
        webhooks = app_data['node']['webhooks']
        assert len(webhooks) == 1
        assert webhooks[0]['name'] in webhooks_names
    assert len(apps_data) == count

def test_apps_with_extensions_query(staff_api_client, permission_manage_apps, permission_manage_orders, app_with_extensions):
    if False:
        return 10
    (app, app_extensions) = app_with_extensions
    response = staff_api_client.post_graphql(QUERY_APPS_WITH_FILTER, permissions=[permission_manage_apps, permission_manage_orders])
    content = get_graphql_content(response)
    apps_data = content['data']['apps']['edges']
    assert len(apps_data) == 1
    app_data = apps_data[0]['node']
    extensions_data = app_data['extensions']
    returned_ids = {e['id'] for e in extensions_data}
    returned_labels = {e['label'] for e in extensions_data}
    returned_mounts = {e['mount'].lower() for e in extensions_data}
    returned_targets = {e['target'].lower() for e in extensions_data}
    returned_permission_codes = [e['permissions'] for e in extensions_data]
    for app_extension in app_extensions:
        global_id = graphene.Node.to_global_id('AppExtension', app_extension.id)
        assert global_id in returned_ids
        assert app_extension.label in returned_labels
        assert app_extension.mount in returned_mounts
        assert app_extension.target in returned_targets
        assigned_permissions = [p.codename for p in app_extension.permissions.all()]
        assigned_permissions = [{'code': p.upper()} for p in assigned_permissions]
        assert assigned_permissions in returned_permission_codes

def test_apps_query_no_permission(staff_api_client, permission_manage_users, permission_manage_staff, app):
    if False:
        while True:
            i = 10
    variables = {'filter': {}}
    response = staff_api_client.post_graphql(QUERY_APPS_WITH_FILTER, variables, permissions=[])
    assert_no_permission(response)
    response = staff_api_client.post_graphql(QUERY_APPS_WITH_FILTER, variables, permissions=[permission_manage_users, permission_manage_staff])
    assert_no_permission(response)
QUERY_APPS_WITH_SORT = '\n    query ($sort_by: AppSortingInput!) {\n        apps(first:5, sortBy: $sort_by) {\n                edges{\n                    node{\n                        name\n                    }\n                }\n            }\n        }\n'

@pytest.mark.parametrize(('apps_sort', 'result_order'), [({'field': 'NAME', 'direction': 'ASC'}, ['facebook', 'google']), ({'field': 'NAME', 'direction': 'DESC'}, ['google', 'facebook']), ({'field': 'CREATION_DATE', 'direction': 'ASC'}, ['google', 'facebook']), ({'field': 'CREATION_DATE', 'direction': 'DESC'}, ['facebook', 'google'])])
def test_query_apps_with_sort(apps_sort, result_order, staff_api_client, permission_manage_apps):
    if False:
        return 10
    with freeze_time('2018-05-31 12:00:01'):
        App.objects.create(name='google', is_active=True)
    with freeze_time('2019-05-31 12:00:01'):
        App.objects.create(name='facebook', is_active=True)
    variables = {'sort_by': apps_sort}
    staff_api_client.user.user_permissions.add(permission_manage_apps)
    response = staff_api_client.post_graphql(QUERY_APPS_WITH_SORT, variables)
    content = get_graphql_content(response)
    apps = content['data']['apps']['edges']
    for (order, account_name) in enumerate(result_order):
        assert apps[order]['node']['name'] == account_name
QUERY_APPS = '\nquery {\n    apps(first: 5){\n        edges {\n            node {\n                id\n            }\n        }\n    }\n}\n'

def test_apps_query_pending_installation(staff_api_client, app):
    if False:
        while True:
            i = 10
    app.is_installed = False
    app.save(update_fields=['is_installed'])
    response = staff_api_client.post_graphql(QUERY_APPS)
    content = get_graphql_content(response)
    assert content['data']['apps']['edges'] == []
QUERY_APPS_FOR_FEDERATION = '\n    query GetAppInFederation($representations: [_Any]) {\n        _entities(representations: $representations) {\n            __typename\n            ... on App {\n                id\n                name\n            }\n        }\n    }\n'

def test_query_app_for_federation(staff_api_client, app, permission_manage_apps):
    if False:
        return 10
    app_id = graphene.Node.to_global_id('App', app.pk)
    variables = {'representations': [{'__typename': 'App', 'id': app_id}]}
    response = staff_api_client.post_graphql(QUERY_APPS_FOR_FEDERATION, variables, permissions=[permission_manage_apps], check_no_permissions=False)
    content = get_graphql_content(response)
    assert content['data']['_entities'] == [{'__typename': 'App', 'id': app_id, 'name': app.name}]

def test_query_app_for_federation_without_permission(api_client, app):
    if False:
        i = 10
        return i + 15
    app_id = graphene.Node.to_global_id('App', app.pk)
    variables = {'representations': [{'__typename': 'App', 'id': app_id}]}
    response = api_client.post_graphql(QUERY_APPS_FOR_FEDERATION, variables)
    content = get_graphql_content(response)
    assert content['data']['_entities'] == [None]
QUERY_APPS_AVAILABLE_FOR_STAFF_WITHOUT_MANAGE_APPS = '\n    query{\n        apps(first: 5){\n            edges{\n                node{\n                    id\n                    isActive\n                    permissions{\n                        name\n                        code\n                    }\n                    name\n                    type\n                    aboutApp\n                    dataPrivacy\n                    dataPrivacyUrl\n                    homepageUrl\n                    supportUrl\n                    configurationUrl\n                    appUrl\n                    accessToken\n                    extensions{\n                        id\n                        label\n                        url\n                        mount\n                        target\n                        permissions{\n                            code\n                        }\n                    }\n                }\n            }\n        }\n    }\n'

@freeze_time('2018-05-31 12:00:01')
def test_apps_query_staff_without_permissions(staff_api_client, staff_user, permission_manage_apps, permission_manage_orders, app):
    if False:
        print('Hello World!')
    app.type = AppType.THIRDPARTY
    app.save()
    response = staff_api_client.post_graphql(QUERY_APPS_AVAILABLE_FOR_STAFF_WITHOUT_MANAGE_APPS)
    content = get_graphql_content(response)
    apps_data = content['data']['apps']['edges']
    assert len(apps_data) == 1
    app_data = apps_data[0]['node']
    expected_id = graphene.Node.to_global_id('App', app.id)
    assert app_data['id'] == expected_id
    assert app_data['accessToken'] == create_access_token_for_app(app, staff_user)

def test_apps_query_for_normal_user(user_api_client, permission_manage_users, permission_manage_staff, app):
    if False:
        print('Hello World!')
    response = user_api_client.post_graphql(QUERY_APPS_AVAILABLE_FOR_STAFF_WITHOUT_MANAGE_APPS)
    assert_no_permission(response)
QUERY_APPS_WITH_METADATA = '\n    query{\n        apps(first: 5){\n            edges{\n                node{\n                    id\n                    metadata{\n                        key\n                        value\n                    }\n\n                }\n            }\n        }\n    }\n'

def test_apps_query_with_metadata_staff_user_without_permissions(staff_api_client, staff_user, app):
    if False:
        return 10
    app.type = AppType.THIRDPARTY
    app.store_value_in_metadata({'test': '123'})
    app.save()
    response = staff_api_client.post_graphql(QUERY_APPS_WITH_METADATA)
    assert_no_permission(response)
QUERY_APPS_WITH_METAFIELD = '\n    query{\n        apps(first: 5){\n            edges{\n                node{\n                    id\n                    metafield(key: "test")\n                }\n            }\n        }\n    }\n'

def test_apps_query_with_metafield_staff_user_without_permissions(staff_api_client, staff_user, app):
    if False:
        return 10
    app.type = AppType.THIRDPARTY
    app.store_value_in_metadata({'test': '123'})
    app.save()
    response = staff_api_client.post_graphql(QUERY_APPS_WITH_METAFIELD)
    assert_no_permission(response)
QUERY_APPS_WITH_METAFIELDS = '\n    query{\n        apps(first: 5){\n            edges{\n                node{\n                    id\n                    metafields(keys: ["test"])\n                }\n            }\n        }\n    }\n'

def test_apps_query_with_metafields_staff_user_without_permissions(staff_api_client, staff_user, app):
    if False:
        return 10
    app.type = AppType.THIRDPARTY
    app.store_value_in_metadata({'test': '123'})
    app.save()
    response = staff_api_client.post_graphql(QUERY_APPS_WITH_METAFIELDS)
    assert_no_permission(response)