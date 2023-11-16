"""HACS configuration schema Test Suite."""
from custom_components.hacs.utils.configuration_schema import hacs_config_combined

def test_combined():
    if False:
        i = 10
        return i + 15
    assert isinstance(hacs_config_combined(), dict)