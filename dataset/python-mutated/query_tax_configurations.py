from ...utils import get_graphql_content
TAX_CONFIGURATIONS_QUERY = '\nquery TaxConfigurationsList($first: Int) {\n  taxConfigurations(first: $first) {\n    edges {\n      node {\n        id\n        channel {\n          id\n          slug\n        }\n        displayGrossPrices\n        pricesEnteredWithTax\n        chargeTaxes\n        taxCalculationStrategy\n      }\n    }\n  }\n}\n\n'

def get_tax_configurations(staff_api_client, first=10):
    if False:
        while True:
            i = 10
    variables = {'first': 10}
    response = staff_api_client.post_graphql(TAX_CONFIGURATIONS_QUERY, variables)
    content = get_graphql_content(response)
    return content['data']['taxConfigurations']['edges']