from ...utils import get_graphql_content
TAX_CLASS_CREATE_MUTATION = '\nmutation TaxClassCreate($input: TaxClassCreateInput!) {\n  taxClassCreate(input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    taxClass {\n      id\n      name\n      countries {\n        country {\n          code\n        }\n        rate\n        taxClass {\n          id\n        }\n      }\n    }\n  }\n}\n'

def create_tax_class(staff_api_client, tax_class_name='Test Tax Class', country_rates=None):
    if False:
        for i in range(10):
            print('nop')
    variables = {'input': {'name': tax_class_name, 'createCountryRates': country_rates}}
    response = staff_api_client.post_graphql(TAX_CLASS_CREATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['taxClassCreate']['errors'] == []
    data = content['data']['taxClassCreate']['taxClass']
    assert data['id'] is not None
    return data