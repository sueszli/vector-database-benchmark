import pytest
from .....account.models import Group
from ....tests.utils import get_graphql_content
QUERY_PERMISSION_GROUP_WITH_SORT = '\nquery ($sort_by: PermissionGroupSortingInput!) {\n    permissionGroups(first:5, sortBy: $sort_by) {\n        edges{\n            node{\n                name\n            }\n        }\n    }\n}\n'

@pytest.mark.parametrize(('permission_group_sort', 'result'), [({'field': 'NAME', 'direction': 'ASC'}, ['Add', 'Manage user group.', 'Remove']), ({'field': 'NAME', 'direction': 'DESC'}, ['Remove', 'Manage user group.', 'Add'])])
def test_permission_group_with_sort(permission_group_manage_users, staff_user, permission_manage_staff, staff_api_client, permission_group_sort, result):
    if False:
        i = 10
        return i + 15
    staff_user.user_permissions.add(permission_manage_staff)
    query = QUERY_PERMISSION_GROUP_WITH_SORT
    Group.objects.bulk_create([Group(name='Add'), Group(name='Remove')])
    variables = {'sort_by': permission_group_sort}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['permissionGroups']['edges']
    for (order, group_name) in enumerate(result):
        assert data[order]['node']['name'] == group_name