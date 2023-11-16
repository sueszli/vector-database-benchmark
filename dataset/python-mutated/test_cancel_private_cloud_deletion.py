from unittest import mock
from google.cloud import vmwareengine_v1
from cancel_private_cloud_deletion import cancel_private_cloud_deletion
from cancel_private_cloud_deletion import cancel_private_cloud_deletion_by_full_name

@mock.patch('google.cloud.vmwareengine_v1.VmwareEngineClient')
def test_cancel(mock_client_class):
    if False:
        print('Hello World!')
    mock_client = mock_client_class.return_value
    cancel_private_cloud_deletion_by_full_name('test_name')
    request = vmwareengine_v1.UndeletePrivateCloudRequest()
    request.name = 'test_name'
    mock_client.undelete_private_cloud.assert_called_once_with(request)
    mock_client.undelete_private_cloud.reset_mock()
    cancel_private_cloud_deletion('project_321', 'zone-33', 'cloud-number-nine')
    request.name = 'projects/project_321/locations/zone-33/privateClouds/cloud-number-nine'
    mock_client.undelete_private_cloud.assert_called_once_with(request)