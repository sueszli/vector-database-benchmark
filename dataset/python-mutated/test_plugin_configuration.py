from unittest import mock
from unittest.mock import Mock
import pytest
import requests
from django.core.exceptions import ValidationError
from requests_hardened import HTTPSession
from ..plugin import NPAtobaraiGatewayPlugin

@mock.patch.object(HTTPSession, 'request')
@pytest.mark.parametrize('status_code', [200, 400])
def test_validate_plugin_configuration_valid_credentials(mocked_request, np_atobarai_plugin, status_code):
    if False:
        while True:
            i = 10
    plugin = np_atobarai_plugin()
    response = Mock(spec=requests.Response, status_code=status_code)
    mocked_request.return_value = response
    NPAtobaraiGatewayPlugin.validate_plugin_configuration(plugin)

@mock.patch.object(HTTPSession, 'request')
@pytest.mark.parametrize('status_code', [401, 403])
def test_validate_plugin_configuration_invalid_credentials(mocked_request, np_atobarai_plugin, status_code):
    if False:
        print('Hello World!')
    plugin = np_atobarai_plugin()
    response = Mock(spec=requests.Response, status_code=status_code, request=Mock())
    mocked_request.return_value = response
    with pytest.raises(ValidationError):
        NPAtobaraiGatewayPlugin.validate_plugin_configuration(plugin)

@mock.patch('saleor.payment.gateways.np_atobarai.api_helpers.requests.request')
def test_validate_plugin_configuration_missing_data(mocked_request, np_atobarai_plugin):
    if False:
        return 10
    plugin = np_atobarai_plugin(merchant_code=None, sp_code=None, terminal_id=None)
    response = Mock(spec=requests.Response, status_code=200)
    mocked_request.return_value = response
    with pytest.raises(ValidationError) as excinfo:
        NPAtobaraiGatewayPlugin.validate_plugin_configuration(plugin)
    assert len(excinfo.value.error_dict) == 3

@mock.patch('saleor.payment.gateways.np_atobarai.api_helpers.requests.request')
def test_validate_plugin_configuration_invalid_shipping_company_code(mocked_request, np_atobarai_plugin):
    if False:
        for i in range(10):
            print('nop')
    plugin = np_atobarai_plugin(shipping_company='00')
    response = Mock(spec=requests.Response, status_code=200)
    mocked_request.return_value = response
    with pytest.raises(ValidationError) as excinfo:
        NPAtobaraiGatewayPlugin.validate_plugin_configuration(plugin)
    assert 'shipping_company' in excinfo.value.error_dict

@mock.patch('saleor.payment.gateways.np_atobarai.api_helpers.requests.request')
def test_validate_plugin_configuration_inactive(mocked_request, np_atobarai_plugin):
    if False:
        return 10
    plugin = np_atobarai_plugin(active=False)
    NPAtobaraiGatewayPlugin.validate_plugin_configuration(plugin)
    mocked_request.assert_not_called()