from ...utils import get_graphql_content
TAX_CONFIGURATION_UPDATE_MUTATION = '\nmutation TaxConfigurationUpdate($id: ID!, $input: TaxConfigurationUpdateInput!) {\n  taxConfigurationUpdate(id: $id, input: $input) {\n    errors {\n      field\n      code\n      message\n      countryCodes\n    }\n    taxConfiguration {\n      id\n      channel {\n        id\n        name\n      }\n      displayGrossPrices\n      pricesEnteredWithTax\n      chargeTaxes\n      taxCalculationStrategy\n      countries {\n        country {\n          code\n        }\n        chargeTaxes\n        taxCalculationStrategy\n        displayGrossPrices\n      }\n    }\n  }\n}\n'

def update_tax_configuration(staff_api_client, tax_config_id, charge_taxes=True, tax_calculation_strategy='FLAT_RATES', display_gross_prices=True, prices_entered_with_tax=True, update_countries_configuration=[], remove_countries_configuration=[]):
    if False:
        for i in range(10):
            print('nop')
    variables = {'id': tax_config_id, 'input': {'chargeTaxes': charge_taxes, 'taxCalculationStrategy': tax_calculation_strategy, 'displayGrossPrices': display_gross_prices, 'pricesEnteredWithTax': prices_entered_with_tax, 'updateCountriesConfiguration': update_countries_configuration, 'removeCountriesConfiguration': remove_countries_configuration}}
    response = staff_api_client.post_graphql(TAX_CONFIGURATION_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['taxConfigurationUpdate']['errors'] == []
    data = content['data']['taxConfigurationUpdate']['taxConfiguration']
    assert data['id'] is not None
    return data