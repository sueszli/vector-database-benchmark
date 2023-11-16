import pytest
from .....menu.models import Menu
from ....tests.utils import get_graphql_content

@pytest.fixture
def menus_for_pagination(db):
    if False:
        while True:
            i = 10
    return Menu.objects.bulk_create([Menu(name='menu1', slug='menu1'), Menu(name='menuMenu1', slug='menuMenu1'), Menu(name='menuMenu2', slug='menuMenu2'), Menu(name='menu2', slug='menu2'), Menu(name='menu3', slug='menu3')])
QUERY_MENUS_PAGINATION = '\n    query (\n        $first: Int, $last: Int, $after: String, $before: String,\n        $sortBy: MenuSortingInput, $filter: MenuFilterInput\n    ){\n        menus(\n            first: $first, last: $last, after: $after, before: $before,\n            sortBy: $sortBy, filter: $filter\n        ) {\n            edges {\n                node {\n                    name\n                }\n            }\n            pageInfo{\n                startCursor\n                endCursor\n                hasNextPage\n                hasPreviousPage\n            }\n        }\n    }\n'

@pytest.mark.parametrize(('sort_by', 'menus_order'), [({'field': 'NAME', 'direction': 'ASC'}, ['footer', 'menu1', 'menu2']), ({'field': 'NAME', 'direction': 'DESC'}, ['navbar', 'menuMenu2', 'menuMenu1'])])
def test_menus_pagination_with_sorting(sort_by, menus_order, staff_api_client, menus_for_pagination):
    if False:
        i = 10
        return i + 15
    page_size = 3
    variables = {'first': page_size, 'after': None, 'sortBy': sort_by}
    response = staff_api_client.post_graphql(QUERY_MENUS_PAGINATION, variables)
    content = get_graphql_content(response)
    menus_nodes = content['data']['menus']['edges']
    assert menus_order[0] == menus_nodes[0]['node']['name']
    assert menus_order[1] == menus_nodes[1]['node']['name']
    assert menus_order[2] == menus_nodes[2]['node']['name']
    assert len(menus_nodes) == page_size

@pytest.mark.parametrize(('filter_by', 'menus_order'), [({'search': 'menuMenu'}, ['menuMenu1', 'menuMenu2']), ({'search': 'menu1'}, ['menu1', 'menuMenu1'])])
def test_menus_pagination_with_filtering(filter_by, menus_order, staff_api_client, menus_for_pagination):
    if False:
        for i in range(10):
            print('nop')
    page_size = 2
    variables = {'first': page_size, 'after': None, 'filter': filter_by}
    response = staff_api_client.post_graphql(QUERY_MENUS_PAGINATION, variables)
    content = get_graphql_content(response)
    menus_nodes = content['data']['menus']['edges']
    assert menus_order[0] == menus_nodes[0]['node']['name']
    assert menus_order[1] == menus_nodes[1]['node']['name']
    assert len(menus_nodes) == page_size