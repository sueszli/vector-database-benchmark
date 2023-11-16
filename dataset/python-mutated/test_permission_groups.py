import pytest
from .....account.models import Group
from ....tests.utils import get_graphql_content

@pytest.fixture
def permission_groups_for_pagination(db):
    if False:
        return 10
    return Group.objects.bulk_create([Group(name='admin'), Group(name='customer_manager'), Group(name='discount_manager'), Group(name='staff'), Group(name='accountant')])
QUERY_PERMISSION_GROUPS_PAGINATION = '\n    query (\n        $first: Int, $last: Int, $after: String, $before: String,\n        $sortBy: PermissionGroupSortingInput, $filter: PermissionGroupFilterInput\n    ){\n        permissionGroups (\n            first: $first, last: $last, after: $after, before: $before,\n            sortBy: $sortBy, filter: $filter\n        ) {\n            edges {\n                node {\n                    name\n                }\n            }\n            pageInfo{\n                startCursor\n                endCursor\n                hasNextPage\n                hasPreviousPage\n            }\n        }\n    }\n'

@pytest.mark.parametrize(('sort_by', 'permission_groups_order'), [({'field': 'NAME', 'direction': 'ASC'}, ['accountant', 'admin', 'customer_manager']), ({'field': 'NAME', 'direction': 'DESC'}, ['staff', 'discount_manager', 'customer_manager'])])
def test_permission_groups_pagination_with_sorting(sort_by, permission_groups_order, staff_api_client, permission_manage_staff, permission_groups_for_pagination):
    if False:
        for i in range(10):
            print('nop')
    page_size = 3
    variables = {'first': page_size, 'after': None, 'sortBy': sort_by}
    response = staff_api_client.post_graphql(QUERY_PERMISSION_GROUPS_PAGINATION, variables, permissions=[permission_manage_staff])
    content = get_graphql_content(response)
    permission_groups_nodes = content['data']['permissionGroups']['edges']
    assert permission_groups_order[0] == permission_groups_nodes[0]['node']['name']
    assert permission_groups_order[1] == permission_groups_nodes[1]['node']['name']
    assert permission_groups_order[2] == permission_groups_nodes[2]['node']['name']
    assert len(permission_groups_nodes) == page_size

def test_permission_groups_pagination_with_filtering(staff_api_client, permission_manage_staff, permission_groups_for_pagination):
    if False:
        i = 10
        return i + 15
    page_size = 2
    variables = {'first': page_size, 'after': None, 'filter': {'search': 'manager'}}
    response = staff_api_client.post_graphql(QUERY_PERMISSION_GROUPS_PAGINATION, variables, permissions=[permission_manage_staff])
    content = get_graphql_content(response)
    permission_groups_nodes = content['data']['permissionGroups']['edges']
    assert permission_groups_nodes[0]['node']['name'] == 'customer_manager'
    assert permission_groups_nodes[1]['node']['name'] == 'discount_manager'
    assert len(permission_groups_nodes) == page_size