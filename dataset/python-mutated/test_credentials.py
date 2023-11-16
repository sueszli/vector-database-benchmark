from unittest import mock
from nsx_credentials import get_nsx_credentials
from vcenter_credentials import get_vcenter_credentials

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_vcredentials(mock_client_class):
    if False:
        return 10
    mock_client = mock_client_class.return_value
    get_vcenter_credentials('p2', 'rrr', 'cname')
    mock_client.show_vcenter_credentials.assert_called_once()
    assert len(mock_client.show_vcenter_credentials.call_args[1]) == 1
    name = mock_client.show_vcenter_credentials.call_args[1]['private_cloud']
    assert name == 'projects/p2/locations/rrr/privateClouds/cname'

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_nsx_credentials(mock_client_class):
    if False:
        for i in range(10):
            print('nop')
    mock_client = mock_client_class.return_value
    get_nsx_credentials('p3', 'rrrr', 'cname')
    mock_client.show_nsx_credentials.assert_called_once()
    assert len(mock_client.show_nsx_credentials.call_args[1]) == 1
    name = mock_client.show_nsx_credentials.call_args[1]['private_cloud']
    assert name == 'projects/p3/locations/rrrr/privateClouds/cname'