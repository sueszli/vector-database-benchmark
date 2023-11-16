from saleor.tax.models import TaxClassCountryRate
from ....tests.utils import assert_no_permission, get_graphql_content
from ..fragments import TAX_COUNTRY_CONFIGURATION_FRAGMENT
QUERY = '\n    query TaxCountryConfigurations {\n        taxCountryConfigurations {\n            ...TaxCountryConfiguration\n        }\n    }\n    ' + TAX_COUNTRY_CONFIGURATION_FRAGMENT

def _test_field_resolvers(data: dict):
    if False:
        print('Hello World!')
    configured_countries = set(TaxClassCountryRate.objects.values_list('country'))
    assert len(data['taxCountryConfigurations']) == len(configured_countries)

def test_tax_country_configurations_query_no_permissions(user_api_client):
    if False:
        for i in range(10):
            print('nop')
    response = user_api_client.post_graphql(QUERY, {}, permissions=[])
    assert_no_permission(response)

def test_tax_country_configurations_query_staff_user(staff_api_client):
    if False:
        i = 10
        return i + 15
    response = staff_api_client.post_graphql(QUERY, {})
    content = get_graphql_content(response)
    _test_field_resolvers(content['data'])

def test_tax_country_configurations_query_app(app_api_client):
    if False:
        while True:
            i = 10
    response = app_api_client.post_graphql(QUERY, {})
    content = get_graphql_content(response)
    _test_field_resolvers(content['data'])