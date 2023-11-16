from google.api_core import operation
from google.cloud import vmwareengine_v1

def cancel_private_cloud_deletion_by_full_name(cloud_name: str) -> operation.Operation:
    if False:
        while True:
            i = 10
    '\n    Cancels in progress deletion of VMWare Private Cloud.\n\n    Args:\n        cloud_name: identifier of the Private Cloud you want to cancel deletion for.\n            Expected format:\n            projects/{project_name}/locations/{zone}/privateClouds/{cloud}\n\n    Returns:\n        An Operation object related to canceling private cloud deletion operation.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.UndeletePrivateCloudRequest()
    request.name = cloud_name
    return client.undelete_private_cloud(request)

def cancel_private_cloud_deletion(project_id: str, zone: str, cloud_name: str) -> operation.Operation:
    if False:
        while True:
            i = 10
    '\n    Cancels in progress deletion of VMWare Private Cloud.\n\n    Args:\n        project_id: name of the project hosting the private cloud.\n        zone: zone in which the private cloud is located in.\n        cloud_name: name of the private cloud to cancel deletion for.\n\n    Returns:\n        An Operation object related to canceling private cloud deletion operation.\n    '
    return cancel_private_cloud_deletion_by_full_name(f'projects/{project_id}/locations/{zone}/privateClouds/{cloud_name}')