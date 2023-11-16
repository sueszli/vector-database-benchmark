import warnings
import pytest
from ..gateways.utils import get_supported_currencies
from ..interface import GatewayConfig

@pytest.fixture
def gateway_config():
    if False:
        for i in range(10):
            print('nop')
    return GatewayConfig(gateway_name='Dummy', auto_capture=True, supported_currencies='', connection_params={'secret-key': 'dummy'})

@pytest.mark.parametrize(('supported_currencies', 'expected_currencies'), [('PLN, USD, EUR', ['PLN', 'USD', 'EUR']), ('PLN,EUR', ['PLN', 'EUR']), (' PLN,EUR ', ['PLN', 'EUR']), ('USD', ['USD'])])
def test_get_supported_currencies(supported_currencies, expected_currencies, gateway_config):
    if False:
        for i in range(10):
            print('nop')
    gateway_config.supported_currencies = supported_currencies
    currencies = get_supported_currencies(gateway_config, 'Test')
    assert currencies == expected_currencies

def test_get_supported_currencies_not_configured(gateway_config):
    if False:
        while True:
            i = 10
    with warnings.catch_warnings(record=True) as warns:
        currencies = get_supported_currencies(gateway_config, 'Test')
        expected_warning = 'Supported currencies not configured for Test, please configure supported currencies for this gateway.'
        assert any([str(warning.message) == expected_warning for warning in warns])
    assert currencies == []