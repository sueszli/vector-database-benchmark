from unittest import mock
from google.cloud import vmwareengine_v1
import pytest
from create_policy import create_network_policy
from delete_policy import delete_network_policy
from update_policy import update_network_policy

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_create_policy(mock_client_class):
    if False:
        print('Hello World!')
    mock_client = mock_client_class.return_value
    create_network_policy('pro', 'reg', '1.2.3.4/26', False, True)
    mock_client.create_network_policy.assert_called_once()
    assert len(mock_client.create_network_policy.call_args[0]) == 1
    request = mock_client.create_network_policy.call_args[0][0]
    assert isinstance(request, vmwareengine_v1.CreateNetworkPolicyRequest)
    assert request.parent == 'projects/pro/locations/reg'
    assert request.network_policy.edge_services_cidr == '1.2.3.4/26'
    assert request.network_policy.external_ip.enabled is True
    assert request.network_policy.internet_access.enabled is False

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_create_policy_value_error(mock_client_class):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        create_network_policy('pro', 'reg', '1.2.3.4/24', False, False)

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_delete_policy(mock_client_class):
    if False:
        i = 10
        return i + 15
    mock_client = mock_client_class.return_value
    delete_network_policy('projakt', 'regiom')
    mock_client.delete_network_policy.assert_called_once()
    assert len(mock_client.delete_network_policy.call_args[0]) == 0
    assert len(mock_client.delete_network_policy.call_args[1]) == 1
    name = mock_client.delete_network_policy.call_args[1]['name']
    assert name == 'projects/projakt/locations/regiom/networkPolicies/regiom-default'

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_update_policy(mock_client_class):
    if False:
        print('Hello World!')
    mock_client = mock_client_class.return_value
    update_network_policy('project', 'regiono', True, False)
    mock_client.update_network_policy.assert_called_once()
    assert len(mock_client.update_network_policy.call_args[0]) == 1
    request = mock_client.update_network_policy.call_args[0][0]
    assert isinstance(request, vmwareengine_v1.UpdateNetworkPolicyRequest)
    assert request.network_policy.name == 'projects/project/locations/regiono/networkPolicies/regiono-default'
    assert request.network_policy.external_ip.enabled is False
    assert request.network_policy.internet_access.enabled is True