from google.api_core import operation
from google.cloud import vmwareengine_v1

def update_cluster_node_count(project_id: str, zone: str, private_cloud_name: str, cluster_name: str, node_count: int) -> operation.Operation:
    if False:
        for i in range(10):
            print('nop')
    '\n    Modify the number of nodes in a cluster in a private cloud.\n\n    Modifying a cluster is a long-running operation and it may take over an hour.\n\n    Args:\n        project_id: name of the project you want to use.\n        zone: zone in which your private cloud is located.\n        private_cloud_name: name of the private cloud hosting the cluster.\n        cluster_name: name of the cluster.\n        node_count: desired number of nodes in the cluster.\n\n    Returns:\n        An Operation object related to cluster modification operation.\n    '
    if node_count < 3:
        raise RuntimeError('Cluster needs to have at least 3 nodes')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.UpdateClusterRequest()
    request.cluster = vmwareengine_v1.Cluster()
    request.cluster.name = f'projects/{project_id}/locations/{zone}/privateClouds/{private_cloud_name}/clusters/{cluster_name}'
    request.cluster.node_type_configs = {'standard-72': vmwareengine_v1.NodeTypeConfig()}
    request.cluster.node_type_configs['standard-72'].node_count = node_count
    request.update_mask = 'nodeTypeConfigs.*.nodeCount'
    return client.update_cluster(request)