from django_countries import countries
from ..utils import get_countries_codes_list, get_track_inventory_by_default

def test_get_countries_codes_list(shipping_zones):
    if False:
        i = 10
        return i + 15
    all_countries = {country[0] for country in countries}
    countries_list_all = get_countries_codes_list()
    assert countries_list_all == all_countries

def test_get_countries_codes_list_true(shipping_zones):
    if False:
        return 10
    fixture_countries_code_set = {zone.countries[0].code for zone in shipping_zones}
    countries_list_true = get_countries_codes_list(attached_to_shipping_zones=True)
    assert countries_list_true == fixture_countries_code_set

def test_get_countries_codes_list_false(shipping_zones):
    if False:
        while True:
            i = 10
    fixture_countries_code_set = {zone.countries[0].code for zone in shipping_zones}
    countries_list_false = get_countries_codes_list(False)
    assert not any((country in countries_list_false for country in fixture_countries_code_set))

def test_get_track_inventory_by_default(dummy_info):
    if False:
        for i in range(10):
            print('nop')
    result = get_track_inventory_by_default(dummy_info)
    assert result is True or result is None