from unittest import mock
from google.cloud import vmwareengine_v1
from create_legacy_network import create_legacy_network
from delete_legacy_network import delete_legacy_network
from list_networks import list_networks

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_network_create(mock_client_class):
    if False:
        while True:
            i = 10
    mock_client = mock_client_class.return_value
    network_mock = mock_client.create_vmware_engine_network.return_value.result.return_value
    network = create_legacy_network('proooject', 'around_here')
    assert network is network_mock
    mock_client.create_vmware_engine_network.assert_called_once()
    assert len(mock_client.create_vmware_engine_network.call_args[0]) == 1
    request = mock_client.create_vmware_engine_network.call_args[0][0]
    assert request.parent == 'projects/proooject/locations/around_here'
    assert request.vmware_engine_network_id == 'around_here-default'
    assert request.vmware_engine_network.type_ == vmwareengine_v1.VmwareEngineNetwork.Type.LEGACY
    assert request.vmware_engine_network.description == 'Legacy network created using vmwareengine_v1.VmwareEngineNetwork'

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_network_list(mock_client_class):
    if False:
        for i in range(10):
            print('nop')
    mock_client = mock_client_class.return_value
    ret = list_networks('projejejkt', 'reggeregion')
    mock_client.list_vmware_engine_networks.assert_called_once_with(parent='projects/projejejkt/locations/reggeregion')
    assert ret is mock_client.list_vmware_engine_networks.return_value

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_network_delete(mock_client_class):
    if False:
        for i in range(10):
            print('nop')
    mock_client = mock_client_class.return_value
    delete_legacy_network('p1', 'r1')
    mock_client.delete_vmware_engine_network.assert_called_once_with(name='projects/p1/locations/r1/vmwareEngineNetworks/r1-default')