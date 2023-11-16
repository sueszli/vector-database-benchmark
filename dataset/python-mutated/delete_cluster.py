from google.api_core import operation
from google.cloud import vmwareengine_v1

def delete_cluster(project_id: str, zone: str, private_cloud_name: str, cluster_name: str) -> operation.Operation:
    if False:
        for i in range(10):
            print('nop')
    '\n    Delete a cluster from private cloud.\n\n    Deleting a cluster is a long-running operation and it may take over an hour..\n\n    Args:\n        project_id: name of the project you want to use.\n        zone: region in which your private cloud is located.\n        private_cloud_name: name of the private cloud hosting the new cluster.\n        cluster_name: name of the new cluster.\n\n    Returns:\n        An Operation object related to started cluster deletion operation.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.DeleteClusterRequest()
    request.name = f'projects/{project_id}/locations/{zone}/privateClouds/{private_cloud_name}/clusters/{cluster_name}'
    return client.delete_cluster(request)