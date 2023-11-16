from google.api_core import operation
from google.cloud import vmwareengine_v1

def create_cluster(project_id: str, zone: str, private_cloud_name: str, cluster_name: str, node_count: int=4) -> operation.Operation:
    if False:
        i = 10
        return i + 15
    '\n    Create a new cluster in a private cloud.\n\n    Creation of a new cluster is a long-running operation and it may take over an hour.\n\n    Args:\n        project_id: name of the project you want to use.\n        zone: region in which your private cloud is located.\n        private_cloud_name: name of the private cloud hosting the new cluster.\n        cluster_name: name of the new cluster.\n        node_count: number of nodes in the new cluster. (Must be >= 3)\n\n    Returns:\n        An Operation object related to started cluster creation operation.\n\n    Raises:\n        ValueError in case an incorrect number of nodes is provided.\n    '
    if node_count < 3:
        raise ValueError('Cluster needs to have at least 3 nodes')
    request = vmwareengine_v1.CreateClusterRequest()
    request.parent = f'projects/{project_id}/locations/{zone}/privateClouds/{private_cloud_name}'
    request.cluster = vmwareengine_v1.Cluster()
    request.cluster.name = cluster_name
    request.cluster.node_type_configs = {'standard-72': vmwareengine_v1.NodeTypeConfig()}
    request.cluster.node_type_configs['standard-72'].node_count = node_count
    client = vmwareengine_v1.VmwareEngineClient()
    return client.create_cluster(request)