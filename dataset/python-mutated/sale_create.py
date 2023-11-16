from ...utils import get_graphql_content
SALE_CREATE_MUTATION = '\nmutation createSale($input: SaleInput!) {\n  saleCreate(input: $input) {\n    errors {\n      field\n      code\n      message\n    }\n    sale {\n      id\n      name\n      type\n      startDate\n      endDate\n    }\n  }\n}\n'

def create_sale(staff_api_client, name='Test sale', sale_type='FIXED'):
    if False:
        for i in range(10):
            print('nop')
    variables = {'input': {'name': name, 'type': sale_type}}
    response = staff_api_client.post_graphql(SALE_CREATE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    assert content['data']['saleCreate']['errors'] == []
    data = content['data']['saleCreate']['sale']
    assert data['id'] is not None
    assert data['name'] == name
    assert data['type'] == sale_type
    return data