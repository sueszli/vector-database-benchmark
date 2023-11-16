from google.api_core import operation
from google.cloud import vmwareengine_v1
DEFAULT_MANAGEMENT_CIDR = '192.168.0.0/24'
DEFAULT_NODE_COUNT = 3

def create_private_cloud(project_id: str, zone: str, network_name: str, cloud_name: str, cluster_name: str) -> operation.Operation:
    if False:
        while True:
            i = 10
    '\n    Creates a new Private Cloud using VMWare Engine.\n\n    Creating a new Private Cloud is a long-running operation and it may take over an hour.\n\n    Args:\n        project_id: name of the project you want to use.\n        zone: the zone you want to use, i.e. "us-central1-a"\n        network_name: name of the VMWareNetwork to use for the new Private Cloud\n        cloud_name: name of the new Private Cloud\n        cluster_name: name for the new cluster in this Private Cloud\n\n    Returns:\n        An operation object representing the started operation. You can call its .result() method to wait for it to finish.\n    '
    request = vmwareengine_v1.CreatePrivateCloudRequest()
    request.parent = f'projects/{project_id}/locations/{zone}'
    request.private_cloud_id = cloud_name
    request.private_cloud = vmwareengine_v1.PrivateCloud()
    request.private_cloud.management_cluster = vmwareengine_v1.PrivateCloud.ManagementCluster()
    request.private_cloud.management_cluster.cluster_id = cluster_name
    node_config = vmwareengine_v1.NodeTypeConfig()
    node_config.node_count = DEFAULT_NODE_COUNT
    request.private_cloud.management_cluster.node_type_configs = {'standard-72': node_config}
    request.private_cloud.network_config = vmwareengine_v1.NetworkConfig()
    request.private_cloud.network_config.vmware_engine_network = network_name
    request.private_cloud.network_config.management_cidr = DEFAULT_MANAGEMENT_CIDR
    client = vmwareengine_v1.VmwareEngineClient()
    return client.create_private_cloud(request)