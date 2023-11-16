from google.api_core import operation
from google.cloud import vmwareengine_v1

def delete_private_cloud_by_full_name(cloud_name: str) -> operation.Operation:
    if False:
        for i in range(10):
            print('nop')
    '\n    Deletes VMWare Private Cloud.\n\n    Args:\n        cloud_name: identifier of the Private Cloud you want to delete.\n            Expected format:\n            projects/{project_name}/locations/{zone}/privateClouds/{cloud}\n\n    Returns:\n        An Operation object related to started private cloud deletion operation.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.DeletePrivateCloudRequest()
    request.force = True
    request.delay_hours = 3
    request.name = cloud_name
    return client.delete_private_cloud(request)

def delete_private_cloud(project_id: str, zone: str, cloud_name: str) -> operation.Operation:
    if False:
        print('Hello World!')
    '\n    Deletes VMWare Private Cloud.\n\n    Args:\n        project_id: name of the project hosting the private cloud.\n        zone: zone in which the private cloud is located in.\n        cloud_name: name of the private cloud to be deleted.\n\n    Returns:\n        An Operation object related to started private cloud deletion operation.\n    '
    return delete_private_cloud_by_full_name(f'projects/{project_id}/locations/{zone}/privateClouds/{cloud_name}')