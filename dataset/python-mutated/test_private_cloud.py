from unittest import mock
import uuid
from create_private_cloud import create_private_cloud
from delete_private_cloud import delete_private_cloud_by_full_name

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_private_cloud_create(mock_client_class):
    if False:
        print('Hello World!')
    mock_client = mock_client_class.return_value
    cloud_name = 'test-cloud-' + uuid.uuid4().hex[:6]
    create_private_cloud('projekto', 'regiono', 'networko', cloud_name, 'management-cluster')
    mock_client.create_private_cloud.assert_called_once()
    assert len(mock_client.create_private_cloud.call_args[0]) == 1
    assert len(mock_client.create_private_cloud.call_args[1]) == 0
    request = mock_client.create_private_cloud.call_args[0][0]
    assert request.private_cloud.management_cluster.cluster_id == 'management-cluster'
    assert request.parent == 'projects/projekto/locations/regiono'
    assert request.private_cloud.network_config.vmware_engine_network == 'networko'

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_delete_cloud_create(mock_client_class):
    if False:
        print('Hello World!')
    mock_client = mock_client_class.return_value
    delete_private_cloud_by_full_name('the_full_name_of_the_cloud')
    mock_client.delete_private_cloud.assert_called_once()
    assert len(mock_client.delete_private_cloud.call_args[0]) == 1
    assert len(mock_client.delete_private_cloud.call_args[1]) == 0
    request = mock_client.delete_private_cloud.call_args[0][0]
    assert request.name == 'the_full_name_of_the_cloud'