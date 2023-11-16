from ....tests.utils import assert_no_permission, get_graphql_content
from ..fragments import TAX_CLASS_FRAGMENT
MUTATION = '\n    mutation TaxClassCreate($input: TaxClassCreateInput!) {\n        taxClassCreate(input: $input) {\n            errors {\n                field\n                message\n                code\n                countryCodes\n            }\n            taxClass {\n                ...TaxClass\n            }\n        }\n    }\n' + TAX_CLASS_FRAGMENT

def _test_no_permissions(api_client):
    if False:
        while True:
            i = 10
    variables = {'input': {'name': 'Test'}}
    response = api_client.post_graphql(MUTATION, variables, permissions=[])
    assert_no_permission(response)

def test_no_permission_staff(staff_api_client):
    if False:
        print('Hello World!')
    _test_no_permissions(staff_api_client)

def test_no_permission_app(app_api_client):
    if False:
        i = 10
        return i + 15
    _test_no_permissions(app_api_client)

def _test_tax_class_create(api_client, permission_manage_taxes):
    if False:
        for i in range(10):
            print('nop')
    name = 'New tax class'
    rate = 23
    country_code = 'PL'
    variables = {'input': {'name': name, 'createCountryRates': [{'countryCode': country_code, 'rate': rate}]}}
    response = api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxClassCreate']
    assert not data['errors']
    assert data['taxClass']['name'] == name
    assert len(data['taxClass']['countries']) == 1
    assert data['taxClass']['countries'][0]['rate'] == rate
    assert data['taxClass']['countries'][0]['country']['code'] == country_code

def test_create_as_staff(staff_api_client, permission_manage_taxes):
    if False:
        while True:
            i = 10
    _test_tax_class_create(staff_api_client, permission_manage_taxes)

def test_create_as_app(app_api_client, permission_manage_taxes):
    if False:
        while True:
            i = 10
    _test_tax_class_create(app_api_client, permission_manage_taxes)