import graphene
import pytest
from .....account.models import Group
from ....tests.utils import assert_no_permission, get_graphql_content
QUERY_PERMISSION_GROUP_WITH_FILTER = '\nquery ($filter: PermissionGroupFilterInput ){\n    permissionGroups(first: 5, filter: $filter){\n        edges{\n            node{\n                id\n                name\n                permissions{\n                    name\n                    code\n                }\n                users {\n                    email\n                }\n                userCanManage\n            }\n        }\n    }\n}\n'

@pytest.mark.parametrize(('permission_group_filter', 'count'), [({'search': 'Manage user groups'}, 1), ({'search': 'Manage'}, 2), ({}, 3)])
def test_permission_groups_query(permission_group_manage_users, staff_user, permission_manage_staff, staff_api_client, permission_group_filter, count):
    if False:
        i = 10
        return i + 15
    staff_user.user_permissions.add(permission_manage_staff)
    query = QUERY_PERMISSION_GROUP_WITH_FILTER
    Group.objects.bulk_create([Group(name='Manage product.'), Group(name='Remove product.')])
    variables = {'filter': permission_group_filter}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['permissionGroups']['edges']
    assert len(data) == count

def test_permission_groups_query_with_filter_by_ids(permission_group_manage_users, permission_manage_staff, staff_api_client):
    if False:
        for i in range(10):
            print('nop')
    query = QUERY_PERMISSION_GROUP_WITH_FILTER
    variables = {'filter': {'ids': [graphene.Node.to_global_id('Group', permission_group_manage_users.pk)]}}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_staff])
    content = get_graphql_content(response)
    data = content['data']['permissionGroups']['edges']
    assert len(data) == 1

def test_permission_groups_no_permission_to_perform(permission_group_manage_users, permission_manage_staff, staff_api_client):
    if False:
        print('Hello World!')
    query = QUERY_PERMISSION_GROUP_WITH_FILTER
    variables = {'filter': {'search': 'Manage user groups'}}
    response = staff_api_client.post_graphql(query, variables)
    assert_no_permission(response)