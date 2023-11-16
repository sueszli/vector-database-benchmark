import graphene
import pytest
from .....tax.error_codes import TaxCountryConfigurationUpdateErrorCode
from .....tax.models import TaxClass, TaxClassCountryRate
from ....tests.utils import assert_no_permission, get_graphql_content
from ..fragments import TAX_COUNTRY_CONFIGURATION_FRAGMENT
MUTATION = '\n    mutation TaxCountryConfigurationUpdate(\n        $countryCode: CountryCode!\n        $updateTaxClassRates: [TaxClassRateInput!]!\n    ) {\n        taxCountryConfigurationUpdate(\n            countryCode: $countryCode\n            updateTaxClassRates: $updateTaxClassRates\n        ) {\n            errors {\n                field\n                message\n                code\n                taxClassIds\n            }\n            taxCountryConfiguration {\n                ...TaxCountryConfiguration\n            }\n        }\n    }\n    ' + TAX_COUNTRY_CONFIGURATION_FRAGMENT

def _test_no_permissions(api_client):
    if False:
        i = 10
        return i + 15
    country_code = 'PL'
    response = api_client.post_graphql(MUTATION, {'countryCode': country_code, 'updateTaxClassRates': []}, permissions=[])
    assert_no_permission(response)

def test_no_permission_staff(staff_api_client):
    if False:
        for i in range(10):
            print('nop')
    _test_no_permissions(staff_api_client)

def test_no_permission_app(app_api_client):
    if False:
        return 10
    _test_no_permissions(app_api_client)

def _test_country_rates_update(api_client, permission_manage_taxes):
    if False:
        while True:
            i = 10
    tax_class_1 = TaxClass.objects.create(name='Books')
    tax_class_2 = TaxClass.objects.create(name='Accessories')
    tax_class_1.country_rates.create(country='PL', rate=23)
    id_1 = graphene.Node.to_global_id('TaxClass', tax_class_1.pk)
    id_2 = graphene.Node.to_global_id('TaxClass', tax_class_2.pk)
    variables = {'countryCode': 'PL', 'updateTaxClassRates': [{'taxClassId': id_1, 'rate': 0}, {'taxClassId': id_2, 'rate': 20}]}
    response = api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxCountryConfigurationUpdate']
    assert not data['errors']
    assert len(data['taxCountryConfiguration']['taxClassCountryRates']) == 2
    response_data = []
    for item in data['taxCountryConfiguration']['taxClassCountryRates']:
        response_data.append({'rate': item['rate'], 'id': item['taxClass']['id']})
    assert {'rate': 0, 'id': id_1} in response_data
    assert {'rate': 20, 'id': id_2} in response_data

def test_update_rates_as_staff(staff_api_client, permission_manage_taxes):
    if False:
        for i in range(10):
            print('nop')
    _test_country_rates_update(staff_api_client, permission_manage_taxes)

def test_update_rates_as_app(app_api_client, permission_manage_taxes):
    if False:
        print('Hello World!')
    _test_country_rates_update(app_api_client, permission_manage_taxes)

def test_create_country_rate_ignore_input_item_when_rate_is_none(staff_api_client, permission_manage_taxes):
    if False:
        while True:
            i = 10
    tax_class_1 = TaxClass.objects.create(name='Books')
    tax_class_2 = TaxClass.objects.create(name='Accessories')
    tax_class_1.country_rates.create(country='PL', rate=23)
    id_1 = graphene.Node.to_global_id('TaxClass', tax_class_1.pk)
    id_2 = graphene.Node.to_global_id('TaxClass', tax_class_2.pk)
    rate_1 = 20
    variables = {'countryCode': 'PL', 'updateTaxClassRates': [{'taxClassId': id_1, 'rate': rate_1}, {'taxClassId': id_2}]}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxCountryConfigurationUpdate']
    assert not data['errors']
    assert len(data['taxCountryConfiguration']['taxClassCountryRates']) == 1
    assert data['taxCountryConfiguration']['taxClassCountryRates'][0]['taxClass']['id'] == id_1
    assert data['taxCountryConfiguration']['taxClassCountryRates'][0]['rate'] == rate_1

def test_delete_country_rate(staff_api_client, permission_manage_taxes):
    if False:
        for i in range(10):
            print('nop')
    tax_class_1 = TaxClass.objects.create(name='Books')
    tax_class_1.country_rates.create(country='PL', rate=23)
    id_1 = graphene.Node.to_global_id('TaxClass', tax_class_1.pk)
    variables = {'countryCode': 'PL', 'updateTaxClassRates': [{'taxClassId': id_1}]}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxCountryConfigurationUpdate']
    assert not data['errors']
    assert len(data['taxCountryConfiguration']['taxClassCountryRates']) == 0

def test_tax_class_id_not_found(staff_api_client, permission_manage_taxes):
    if False:
        print('Hello World!')
    TaxClass.objects.create(name='Books')
    id = 'spanishinquisition'
    variables = {'countryCode': 'PL', 'updateTaxClassRates': [{'taxClassId': id, 'rate': 20}]}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxCountryConfigurationUpdate']
    assert data['errors']
    assert data['errors'][0]['code'] == TaxCountryConfigurationUpdateErrorCode.NOT_FOUND.name
    assert data['errors'][0]['taxClassIds'] == [id]

@pytest.mark.parametrize('rate', [0, 23])
def test_update_default_country_rate(staff_api_client, permission_manage_taxes, rate):
    if False:
        while True:
            i = 10
    variables = {'countryCode': 'PL', 'updateTaxClassRates': [{'rate': rate}]}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxCountryConfigurationUpdate']
    assert not data['errors']
    assert len(data['taxCountryConfiguration']['taxClassCountryRates']) == 1
    response_data = []
    for item in data['taxCountryConfiguration']['taxClassCountryRates']:
        response_data.append({'rate': item['rate'], 'id': None})
    assert {'rate': rate, 'id': None} in response_data

def test_delete_default_country_rate(staff_api_client, permission_manage_taxes):
    if False:
        i = 10
        return i + 15
    default_rate = TaxClassCountryRate.objects.create(country='PL', rate=0)
    variables = {'countryCode': 'PL', 'updateTaxClassRates': [{'rate': None}]}
    staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    with pytest.raises(TaxClassCountryRate.DoesNotExist):
        default_rate.refresh_from_db()

def test_update_default_country_rate_throws_error_with_multiple_rates(staff_api_client, permission_manage_taxes):
    if False:
        while True:
            i = 10
    variables = {'countryCode': 'PL', 'updateTaxClassRates': [{'rate': 23}, {'rate': 25}]}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxCountryConfigurationUpdate']
    assert data['errors']
    code = TaxCountryConfigurationUpdateErrorCode.ONLY_ONE_DEFAULT_COUNTRY_RATE_ALLOWED
    assert data['errors'][0]['code'] == code.name

def test_validate_negative_rates(staff_api_client, permission_manage_taxes):
    if False:
        i = 10
        return i + 15
    tax_class_1 = TaxClass.objects.create(name='Books')
    tax_class_2 = TaxClass.objects.create(name='Accessories')
    id_1 = graphene.Node.to_global_id('TaxClass', tax_class_1.pk)
    id_2 = graphene.Node.to_global_id('TaxClass', tax_class_2.pk)
    variables = {'countryCode': 'PL', 'updateTaxClassRates': [{'rate': -1}, {'taxClassId': id_1, 'rate': -1}, {'taxClassId': id_2, 'rate': 0}]}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxCountryConfigurationUpdate']
    assert data['errors']
    code = TaxCountryConfigurationUpdateErrorCode.CANNOT_CREATE_NEGATIVE_RATE
    assert data['errors'][0]['code'] == code.name
    assert len(data['errors'][0]['taxClassIds']) == 1
    assert id_1 in data['errors'][0]['taxClassIds']