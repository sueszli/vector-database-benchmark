import graphene
from .....tax.error_codes import TaxClassUpdateErrorCode
from .....tax.models import TaxClass, TaxClassCountryRate
from ....tests.utils import assert_no_permission, get_graphql_content
from ..fragments import TAX_CLASS_FRAGMENT
MUTATION = '\n    mutation TaxClassUpdate($id: ID!, $input: TaxClassUpdateInput!) {\n        taxClassUpdate(id: $id, input: $input) {\n            errors {\n                field\n                message\n                code\n                countryCodes\n            }\n            taxClass {\n                ...TaxClass\n            }\n        }\n    }\n    ' + TAX_CLASS_FRAGMENT

def _test_no_permissions(api_client):
    if False:
        while True:
            i = 10
    tax_class = TaxClass.objects.first()
    id = graphene.Node.to_global_id('TaxClass', tax_class.pk)
    variables = {'id': id, 'input': {'name': 'Test'}}
    response = api_client.post_graphql(MUTATION, variables, permissions=[])
    assert_no_permission(response)

def test_no_permission_staff(staff_api_client):
    if False:
        print('Hello World!')
    _test_no_permissions(staff_api_client)

def test_no_permission_app(app_api_client):
    if False:
        return 10
    _test_no_permissions(app_api_client)

def _test_tax_class_update(api_client, permission_manage_taxes):
    if False:
        i = 10
        return i + 15
    tax_class = TaxClass.objects.create(name='Tax Class')
    tax_class.country_rates.create(country='PL', rate=21)
    id = graphene.Node.to_global_id('TaxClass', tax_class.pk)
    new_name = 'New tax class name'
    update_PL_rate = {'countryCode': 'PL', 'rate': 23}
    create_DE_rate = {'countryCode': 'DE', 'rate': 19}
    variables = {'id': id, 'input': {'name': new_name, 'updateCountryRates': [update_PL_rate, create_DE_rate]}}
    response = api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxClassUpdate']
    assert not data['errors']
    assert data['taxClass']['name'] == new_name
    assert len(data['taxClass']['countries']) == 2
    response_data = []
    for item in data['taxClass']['countries']:
        new_item = {**item, 'countryCode': item['country']['code']}
        new_item.pop('country')
        response_data.append(new_item)
    assert update_PL_rate in response_data
    assert create_DE_rate in response_data

def test_create_as_staff(staff_api_client, permission_manage_taxes):
    if False:
        return 10
    _test_tax_class_update(staff_api_client, permission_manage_taxes)

def test_create_as_app(app_api_client, permission_manage_taxes):
    if False:
        i = 10
        return i + 15
    _test_tax_class_update(app_api_client, permission_manage_taxes)

def test_raise_duplicated_item_error(staff_api_client, permission_manage_taxes):
    if False:
        while True:
            i = 10
    tax_class = TaxClass.objects.create(name='Tax Class')
    tax_class.country_rates.create(country='PL', rate=21)
    id = graphene.Node.to_global_id('TaxClass', tax_class.pk)
    variables = {'id': id, 'input': {'updateCountryRates': [{'countryCode': 'PL', 'rate': 23}], 'removeCountryRates': ['PL']}}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    errors = content['data']['taxClassUpdate']['errors']
    assert errors
    assert errors[0]['code'] == TaxClassUpdateErrorCode.DUPLICATED_INPUT_ITEM.name
    assert errors[0]['countryCodes'] == ['PL']

def test_remove_all_country_rates(staff_api_client, permission_manage_taxes):
    if False:
        return 10
    tax_class = TaxClass.objects.create(name='Tax Class')
    tax_class.country_rates.create(country='PL', rate=21)
    id = graphene.Node.to_global_id('TaxClass', tax_class.pk)
    variables = {'id': id, 'input': {'removeCountryRates': ['PL']}}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxClassUpdate']
    assert not data['errors']
    assert len(data['taxClass']['countries']) == 0

def test_remove_individual_country_rates(staff_api_client, permission_manage_taxes):
    if False:
        i = 10
        return i + 15
    (default_rate_de, _) = TaxClassCountryRate.objects.get_or_create(country='DE', tax_class=None, rate=19)
    tax_class = TaxClass.objects.create(name='Tax Class')
    tax_class.country_rates.create(country='PL', rate=21)
    tax_class.country_rates.create(country='DE', rate=19)
    id = graphene.Node.to_global_id('TaxClass', tax_class.pk)
    new_rate_pl = 23
    variables = {'id': id, 'input': {'updateCountryRates': [{'countryCode': 'PL', 'rate': new_rate_pl}, {'countryCode': 'DE'}]}}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxClassUpdate']
    assert not data['errors']
    assert len(data['taxClass']['countries']) == 1
    assert data['taxClass']['countries'][0]['country']['code'] == 'PL'
    assert data['taxClass']['countries'][0]['rate'] == new_rate_pl
    default_rate_de.refresh_from_db()
    assert default_rate_de.pk is not None

def test_remove_individual_country_rates_non_existing_rate(staff_api_client, permission_manage_taxes):
    if False:
        for i in range(10):
            print('nop')
    (default_rate_de, _) = TaxClassCountryRate.objects.get_or_create(country='DE', tax_class=None, rate=19)
    tax_class = TaxClass.objects.create(name='Tax Class')
    tax_class.country_rates.create(country='PL', rate=21)
    id = graphene.Node.to_global_id('TaxClass', tax_class.pk)
    new_rate_pl = 23
    variables = {'id': id, 'input': {'updateCountryRates': [{'countryCode': 'PL', 'rate': new_rate_pl}, {'countryCode': 'DE'}]}}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxClassUpdate']
    assert not data['errors']
    assert len(data['taxClass']['countries']) == 1
    assert data['taxClass']['countries'][0]['country']['code'] == 'PL'
    default_rate_de.refresh_from_db()
    assert default_rate_de.pk is not None

def test_tax_class_update_zero_rate(staff_api_client, permission_manage_taxes):
    if False:
        i = 10
        return i + 15
    tax_class = TaxClass.objects.create(name='Tax Class')
    tax_class.country_rates.create(country='PL', rate=21)
    id = graphene.Node.to_global_id('TaxClass', tax_class.pk)
    new_name = 'New tax class name'
    update_PL_rate = {'countryCode': 'PL', 'rate': 0}
    variables = {'id': id, 'input': {'name': new_name, 'updateCountryRates': [update_PL_rate]}}
    response = staff_api_client.post_graphql(MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxClassUpdate']
    assert not data['errors']
    assert data['taxClass']['name'] == new_name
    assert len(data['taxClass']['countries']) == 1
    assert data['taxClass']['countries'][0]['rate'] == 0.0