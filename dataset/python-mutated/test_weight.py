import pytest
from measurement.measures import Weight
from ..units import WeightUnits
from ..weight import convert_weight, convert_weight_to_default_weight_unit, get_default_weight_unit

def test_convert_weight():
    if False:
        for i in range(10):
            print('nop')
    weight = Weight(kg=1)
    expected_result = Weight(g=1000)
    result = convert_weight(weight, WeightUnits.G)
    assert result == expected_result

def test_get_default_weight_unit(site_settings):
    if False:
        for i in range(10):
            print('nop')
    result = get_default_weight_unit()
    assert result == site_settings.default_weight_unit

@pytest.mark.parametrize(('default_weight_unit', 'expected_value'), [(WeightUnits.KG, Weight(kg=1)), (WeightUnits.G, Weight(g=1000)), (WeightUnits.LB, Weight(lb=2.205)), (WeightUnits.OZ, Weight(oz=35.274))])
def test_convert_weight_to_default_weight_unit(default_weight_unit, expected_value, site_settings):
    if False:
        i = 10
        return i + 15
    site_settings.default_weight_unit = default_weight_unit
    site_settings.save(update_fields=['default_weight_unit'])
    result = convert_weight_to_default_weight_unit(Weight(kg=1))
    assert result == expected_value