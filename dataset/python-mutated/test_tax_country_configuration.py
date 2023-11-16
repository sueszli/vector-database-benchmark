from typing import TYPE_CHECKING
import graphene
from saleor.tax.models import TaxClassCountryRate
from ....tests.utils import assert_no_permission, get_graphql_content
from ..fragments import TAX_COUNTRY_CONFIGURATION_FRAGMENT
if TYPE_CHECKING:
    from django.db.models import QuerySet
QUERY = '\n    query TaxCountryConfiguration($countryCode: CountryCode!) {\n        taxCountryConfiguration(countryCode: $countryCode) {\n            ...TaxCountryConfiguration\n        }\n    }\n    ' + TAX_COUNTRY_CONFIGURATION_FRAGMENT

def _test_field_resolvers(country_code, country_rates: 'QuerySet[TaxClassCountryRate]', data: dict):
    if False:
        for i in range(10):
            print('nop')
    assert data['country']['code'] == country_code
    assert data['taxClassCountryRates']
    assert len(data['taxClassCountryRates']) == len(country_rates)
    for country_rate in country_rates:
        expected_rate_data = {'rate': country_rate.rate, 'taxClass': {'id': graphene.Node.to_global_id('TaxClass', country_rate.tax_class.pk), 'name': country_rate.tax_class.name}}
        assert expected_rate_data in data['taxClassCountryRates']

def test_tax_country_configuration_query_no_permissions(user_api_client):
    if False:
        return 10
    country_code = 'PL'
    response = user_api_client.post_graphql(QUERY, {'countryCode': country_code}, permissions=[])
    assert_no_permission(response)

def test_tax_country_configuration_query_staff_user(staff_api_client):
    if False:
        i = 10
        return i + 15
    country_code = 'PL'
    country_rates = TaxClassCountryRate.objects.filter(country='PL')
    response = staff_api_client.post_graphql(QUERY, {'countryCode': country_code})
    content = get_graphql_content(response)
    _test_field_resolvers(country_code, country_rates, content['data']['taxCountryConfiguration'])

def test_tax_country_configuration_query_app(app_api_client):
    if False:
        for i in range(10):
            print('nop')
    country_code = 'PL'
    country_rates = TaxClassCountryRate.objects.filter(country='PL')
    response = app_api_client.post_graphql(QUERY, {'countryCode': country_code})
    content = get_graphql_content(response)
    _test_field_resolvers(country_code, country_rates, content['data']['taxCountryConfiguration'])