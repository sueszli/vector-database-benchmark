import graphene
import pytest
from .....account.models import Group
from .....permission.enums import AccountPermissions, OrderPermissions
from ....tests.utils import get_graphql_content

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_permission_group_create(staff_user, permission_manage_staff, staff_api_client, permission_group_manage_users, permission_group_manage_apps, permission_group_manage_staff, count_queries):
    if False:
        while True:
            i = 10
    staff_user.groups.add(permission_group_manage_users, permission_group_manage_apps, permission_group_manage_staff)
    query = '\n        mutation PermissionGroupCreate(\n        $input: PermissionGroupCreateInput!) {\n        permissionGroupCreate(\n            input: $input)\n        {\n            group{\n                id\n                name\n                permissions {\n                    name\n                    code\n                }\n                users {\n                    email\n                }\n            }\n            errors{\n                field\n                code\n                permissions\n                users\n                message\n            }\n        }\n    }\n    '
    group_count = Group.objects.count()
    variables = {'input': {'name': 'New permission group', 'addPermissions': [AccountPermissions.MANAGE_USERS.name], 'addUsers': [graphene.Node.to_global_id('User', staff_user.id)]}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['permissionGroupCreate']
    groups = Group.objects.all()
    assert data['errors'] == []
    assert len(groups) == group_count + 1

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_permission_group_update(permission_group_manage_users, staff_users, permission_manage_staff, staff_api_client, permission_manage_apps, permission_manage_users, permission_manage_orders, count_queries):
    if False:
        i = 10
        return i + 15
    query = '\n    mutation PermissionGroupUpdate(\n        $id: ID!, $input: PermissionGroupUpdateInput!) {\n        permissionGroupUpdate(\n            id: $id, input: $input)\n        {\n            group{\n                id\n                name\n                permissions {\n                    name\n                    code\n                }\n            }\n            errors{\n                field\n                code\n                permissions\n                users\n                message\n            }\n        }\n    }\n    '
    staff_user = staff_users[0]
    staff_user.user_permissions.add(permission_manage_apps, permission_manage_users, permission_manage_orders)
    (group1, group2) = Group.objects.bulk_create([Group(name='manage users'), Group(name='manage staff and users')])
    group1.permissions.add(permission_manage_users)
    group2.permissions.add(permission_manage_users, permission_manage_staff)
    group1_user = staff_users[1]
    group1.user_set.add(group1_user)
    group2.user_set.add(staff_user)
    group_count = Group.objects.count()
    variables = {'id': graphene.Node.to_global_id('Group', group1.id), 'input': {'name': 'New permission group', 'addPermissions': [OrderPermissions.MANAGE_ORDERS.name], 'removePermissions': [AccountPermissions.MANAGE_USERS.name], 'addUsers': [graphene.Node.to_global_id('User', staff_user.pk)], 'removeUsers': [graphene.Node.to_global_id('User', group1.user_set.first().pk)]}}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['permissionGroupUpdate']
    groups = Group.objects.all()
    assert data['errors'] == []
    assert len(groups) == group_count

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_permission_group_update_remove_users_with_manage_staff(permission_group_manage_users, staff_users, permission_manage_staff, staff_api_client, permission_manage_apps, permission_manage_users, permission_manage_orders, count_queries):
    if False:
        for i in range(10):
            print('nop')
    query = '\n    mutation PermissionGroupUpdate(\n        $id: ID!, $input: PermissionGroupUpdateInput!) {\n        permissionGroupUpdate(\n            id: $id, input: $input)\n        {\n            group{\n                id\n                name\n                permissions {\n                    name\n                    code\n                }\n                users {\n                    email\n                }\n            }\n            errors{\n                field\n                code\n                permissions\n                users\n                message\n            }\n        }\n    }\n    '
    (staff_user, staff_user1, staff_user2) = staff_users
    groups = Group.objects.bulk_create([Group(name='manage users'), Group(name='manage staff, order and users')])
    (group1, group2) = groups
    group1.permissions.add(permission_manage_staff, permission_manage_users)
    group2.permissions.add(permission_manage_staff, permission_manage_orders, permission_manage_users)
    group1.user_set.add(staff_user1, staff_user2)
    group2.user_set.add(staff_user2)
    staff_user.user_permissions.add(permission_manage_users, permission_manage_orders)
    variables = {'id': graphene.Node.to_global_id('Group', group1.id), 'input': {'removeUsers': [graphene.Node.to_global_id('User', user.id) for user in [staff_user1, staff_user2]]}}
    response = staff_api_client.post_graphql(query, variables, permissions=(permission_manage_staff,))
    content = get_graphql_content(response)
    data = content['data']['permissionGroupUpdate']
    assert len(data['group']['users']) == 0

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_permission_group_delete(staff_users, permission_manage_staff, permission_manage_orders, permission_manage_products, staff_api_client, count_queries):
    if False:
        i = 10
        return i + 15
    query = '\n    mutation PermissionGroupDelete($id: ID!) {\n        permissionGroupDelete(\n            id: $id)\n        {\n            group{\n                id\n                name\n                permissions {\n                    name\n                    code\n                }\n            }\n            errors{\n                field\n                code\n                users\n                permissions\n                message\n            }\n        }\n    }\n    '
    (staff_user1, staff_user2, _) = staff_users
    staff_user1.user_permissions.add(permission_manage_orders, permission_manage_products)
    groups = Group.objects.bulk_create([Group(name='manage orders'), Group(name='manage orders and products')])
    (group1, group2) = groups
    group1.permissions.add(permission_manage_orders, permission_manage_staff)
    group2.permissions.add(permission_manage_orders, permission_manage_products, permission_manage_staff)
    staff_user2.groups.add(group1, group2)
    group_count = Group.objects.count()
    variables = {'id': graphene.Node.to_global_id('Group', group1.id)}
    response = staff_api_client.post_graphql(query, variables, permissions=(permission_manage_staff,))
    content = get_graphql_content(response)
    data = content['data']['permissionGroupDelete']
    assert data
    assert Group.objects.count() == group_count - 1

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_permission_group_query(permission_group_manage_users, staff_user, permission_manage_staff, permission_manage_users, staff_api_client, count_queries):
    if False:
        for i in range(10):
            print('nop')
    staff_user.user_permissions.add(permission_manage_staff, permission_manage_users)
    group = permission_group_manage_users
    query = '\n    query ($id: ID!){\n        permissionGroup(id: $id){\n            id\n            name\n            permissions {\n                name\n                code\n            }\n            users{\n                email\n            }\n            userCanManage\n        }\n    }\n    '
    variables = {'id': graphene.Node.to_global_id('Group', group.id)}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['permissionGroup']
    assert data

@pytest.mark.django_db
@pytest.mark.count_queries(autouse=False)
def test_groups_for_federation_query_count(api_client, django_assert_num_queries, count_queries):
    if False:
        print('Hello World!')
    groups = Group.objects.bulk_create([Group(name='group 1'), Group(name='group 2'), Group(name='group 3')])
    query = '\n        query GetGroupInFederation($representations: [_Any]) {\n            _entities(representations: $representations) {\n                __typename\n                ... on Group {\n                    id\n                }\n            }\n        }\n    '
    variables = {'representations': [{'__typename': 'Group', 'id': graphene.Node.to_global_id('Group', groups[0].pk)}]}
    with django_assert_num_queries(2):
        response = api_client.post_graphql(query, variables)
        content = get_graphql_content(response)
        assert len(content['data']['_entities']) == 1
    variables = {'representations': [{'__typename': 'Group', 'id': graphene.Node.to_global_id('Group', group.pk)} for group in groups]}
    with django_assert_num_queries(2):
        response = api_client.post_graphql(query, variables)
        content = get_graphql_content(response)
        assert len(content['data']['_entities']) == 3