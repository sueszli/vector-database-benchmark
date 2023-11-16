import re
from ....tests.utils import get_graphql_content
GET_ADDRESS_VALIDATION_RULES_QUERY = '\n    query getValidator(\n        $country_code: CountryCode!, $country_area: String, $city_area: String) {\n        addressValidationRules(\n                countryCode: $country_code,\n                countryArea: $country_area,\n                cityArea: $city_area) {\n            countryCode\n            countryName\n            addressFormat\n            addressLatinFormat\n            allowedFields\n            requiredFields\n            upperFields\n            countryAreaType\n            countryAreaChoices {\n                verbose\n                raw\n            }\n            cityType\n            cityChoices {\n                raw\n                verbose\n            }\n            cityAreaType\n            cityAreaChoices {\n                raw\n                verbose\n            }\n            postalCodeType\n            postalCodeMatchers\n            postalCodeExamples\n            postalCodePrefix\n        }\n    }\n'

def test_address_validation_rules(user_api_client):
    if False:
        for i in range(10):
            print('nop')
    query = GET_ADDRESS_VALIDATION_RULES_QUERY
    variables = {'country_code': 'PL', 'country_area': None, 'city_area': None}
    response = user_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['addressValidationRules']
    assert data['countryCode'] == 'PL'
    assert data['countryName'] == 'POLAND'
    assert data['addressFormat'] is not None
    assert data['addressLatinFormat'] is not None
    assert data['cityType'] == 'city'
    assert data['cityAreaType'] == 'suburb'
    matcher = data['postalCodeMatchers'][0]
    matcher = re.compile(matcher)
    assert matcher.match('00-123')
    assert not data['cityAreaChoices']
    assert not data['cityChoices']
    assert not data['countryAreaChoices']
    assert data['postalCodeExamples']
    assert data['postalCodeType'] == 'postal'
    assert set(data['allowedFields']) == {'companyName', 'city', 'postalCode', 'streetAddress1', 'name', 'streetAddress2'}
    assert set(data['requiredFields']) == {'postalCode', 'streetAddress1', 'city'}
    assert set(data['upperFields']) == {'city'}

def test_address_validation_rules_with_country_area(user_api_client):
    if False:
        while True:
            i = 10
    query = GET_ADDRESS_VALIDATION_RULES_QUERY
    variables = {'country_code': 'CN', 'country_area': 'Fujian Sheng', 'city_area': None}
    response = user_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['addressValidationRules']
    assert data['countryCode'] == 'CN'
    assert data['countryName'] == 'CHINA'
    assert data['countryAreaType'] == 'province'
    assert data['countryAreaChoices']
    assert data['cityType'] == 'city'
    assert data['cityChoices']
    assert data['cityAreaType'] == 'district'
    assert not data['cityAreaChoices']
    assert data['cityChoices']
    assert data['countryAreaChoices']
    assert data['postalCodeExamples']
    assert data['postalCodeType'] == 'postal'
    assert set(data['allowedFields']) == {'city', 'postalCode', 'streetAddress1', 'name', 'streetAddress2', 'countryArea', 'companyName', 'cityArea'}
    assert set(data['requiredFields']) == {'postalCode', 'streetAddress1', 'city', 'countryArea'}
    assert set(data['upperFields']) == {'countryArea'}

def test_address_validation_rules_fields_in_camel_case(user_api_client):
    if False:
        return 10
    query = '\n    query getValidator(\n        $country_code: CountryCode!) {\n        addressValidationRules(countryCode: $country_code) {\n            requiredFields\n            allowedFields\n        }\n    }\n    '
    variables = {'country_code': 'PL'}
    response = user_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    data = content['data']['addressValidationRules']
    required_fields = data['requiredFields']
    allowed_fields = data['allowedFields']
    assert 'streetAddress1' in required_fields
    assert 'streetAddress2' not in required_fields
    assert 'streetAddress1' in allowed_fields
    assert 'streetAddress2' in allowed_fields