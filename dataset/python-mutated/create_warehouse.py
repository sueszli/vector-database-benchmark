from ... import DEFAULT_ADDRESS
from ...utils import get_graphql_content
WAREHOUSE_CREATE_MUTATION = '\nmutation createWarehouse($input: WarehouseCreateInput!) {\n  createWarehouse(input: $input) {\n    errors {\n      message\n      field\n      code\n    }\n    warehouse {\n      id\n      name\n      slug\n    }\n  }\n}\n'

def create_warehouse(staff_api_client, name='Test warehouse', slug='test-slug', address=DEFAULT_ADDRESS):
    if False:
        while True:
            i = 10
    variables = {'input': {'name': name, 'slug': slug, 'address': address}}
    response = staff_api_client.post_graphql(WAREHOUSE_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['createWarehouse']['errors'] == []
    data = content['data']['createWarehouse']['warehouse']
    assert data['id'] is not None
    assert data['name'] == name
    assert data['slug'] == slug
    return data