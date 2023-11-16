from .....tax.models import TaxClass
from ....tests.utils import assert_no_permission, get_graphql_content
from ..fragments import TAX_COUNTRY_CONFIGURATION_FRAGMENT
MUTATION = '\n    mutation TaxCountryConfigurationDelete($countryCode: CountryCode!) {\n        taxCountryConfigurationDelete(countryCode: $countryCode) {\n            errors {\n                field\n                message\n                code\n            }\n            taxCountryConfiguration {\n                ...TaxCountryConfiguration\n            }\n        }\n    }\n    ' + TAX_COUNTRY_CONFIGURATION_FRAGMENT

def _test_no_permissions(api_client):
    if False:
        i = 10
        return i + 15
    country_code = 'PL'
    response = api_client.post_graphql(MUTATION, {'countryCode': country_code}, permissions=[])
    assert_no_permission(response)

def test_no_permission_staff(staff_api_client):
    if False:
        for i in range(10):
            print('nop')
    _test_no_permissions(staff_api_client)

def test_no_permission_app(app_api_client):
    if False:
        while True:
            i = 10
    _test_no_permissions(app_api_client)

def _test_delete_tax_rates_for_country(api_client, permission_manage_taxes):
    if False:
        for i in range(10):
            print('nop')
    country_code = 'PL'
    tax_class_1 = TaxClass.objects.create(name='Books')
    tax_class_2 = TaxClass.objects.create(name='Accessories')
    tax_class_1.country_rates.create(country=country_code, rate=23)
    tax_class_2.country_rates.create(country=country_code, rate=23)
    response = api_client.post_graphql(MUTATION, {'countryCode': country_code}, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxCountryConfigurationDelete']
    assert not data['errors']
    assert len(data['taxCountryConfiguration']['taxClassCountryRates']) == 0

def test_delete_tax_rates_for_country_by_staff(staff_api_client, permission_manage_taxes):
    if False:
        print('Hello World!')
    _test_delete_tax_rates_for_country(staff_api_client, permission_manage_taxes)

def test_delete_tax_rates_for_country_by_app(app_api_client, permission_manage_taxes):
    if False:
        print('Hello World!')
    _test_delete_tax_rates_for_country(app_api_client, permission_manage_taxes)