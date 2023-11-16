import graphene
from ....tests.utils import assert_no_permission, get_graphql_content, get_graphql_content_from_response
QUERY_VOUCHER_BY_ID = '\n    query Voucher($id: ID!) {\n        voucher(id: $id) {\n            id\n            code\n            name\n            discountValue\n        }\n    }\n'

def test_staff_query_voucher(staff_api_client, voucher, permission_manage_discounts):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = staff_api_client.post_graphql(QUERY_VOUCHER_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert content['data']['voucher']['name'] == voucher.name
    assert content['data']['voucher']['code'] == voucher.codes.first().code

def test_query_voucher_by_app(app_api_client, voucher, permission_manage_discounts):
    if False:
        print('Hello World!')
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = app_api_client.post_graphql(QUERY_VOUCHER_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert content['data']['voucher']['name'] == voucher.name
    assert content['data']['voucher']['code'] == voucher.codes.first().code

def test_query_voucher_by_customer(api_client, voucher, permission_manage_discounts):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': graphene.Node.to_global_id('Voucher', voucher.pk)}
    response = api_client.post_graphql(QUERY_VOUCHER_BY_ID, variables)
    assert_no_permission(response)

def test_staff_query_voucher_by_invalid_id(staff_api_client, voucher, permission_manage_discounts):
    if False:
        for i in range(10):
            print('nop')
    id = 'bh/'
    variables = {'id': id}
    response = staff_api_client.post_graphql(QUERY_VOUCHER_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content_from_response(response)
    assert len(content['errors']) == 1
    assert content['errors'][0]['message'] == f'Invalid ID: {id}. Expected: Voucher.'
    assert content['data']['voucher'] is None

def test_staff_query_voucher_with_invalid_object_type(staff_api_client, voucher, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    variables = {'id': graphene.Node.to_global_id('Order', voucher.pk)}
    response = staff_api_client.post_graphql(QUERY_VOUCHER_BY_ID, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    assert content['data']['voucher'] is None