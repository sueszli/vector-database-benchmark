from ...utils import get_graphql_content
TAX_COUNTRY_CONFIGURATION_UPDATE_MUTATION = '\nmutation TaxCountryConfigurationUpdate($countryCode: CountryCode!,\n $updateTaxClassRates: [TaxClassRateInput!]!) {\n  taxCountryConfigurationUpdate(\n    countryCode: $countryCode\n    updateTaxClassRates: $updateTaxClassRates\n  ) {\n    errors {\n      code\n      field\n      message\n      taxClassIds\n    }\n    taxCountryConfiguration {\n      country{\n        code\n      }\n      taxClassCountryRates{\n        country{\n          code\n        }\n        rate\n        taxClass{\n          id\n        }\n      }\n    }\n  }\n}\n\n'

def update_country_tax_rates(staff_api_client, country_code, update_tax_class_rates=[]):
    if False:
        for i in range(10):
            print('nop')
    variables = {'countryCode': country_code, 'updateTaxClassRates': update_tax_class_rates}
    response = staff_api_client.post_graphql(TAX_COUNTRY_CONFIGURATION_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['taxCountryConfigurationUpdate']['errors'] == []
    data = content['data']['taxCountryConfigurationUpdate']['taxCountryConfiguration']
    return data