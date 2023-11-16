from google.cloud import gke_multicloud_v1

def sample_update_azure_node_pool():
    if False:
        i = 10
        return i + 15
    client = gke_multicloud_v1.AzureClustersClient()
    azure_node_pool = gke_multicloud_v1.AzureNodePool()
    azure_node_pool.version = 'version_value'
    azure_node_pool.config.ssh_config.authorized_key = 'authorized_key_value'
    azure_node_pool.subnet_id = 'subnet_id_value'
    azure_node_pool.autoscaling.min_node_count = 1489
    azure_node_pool.autoscaling.max_node_count = 1491
    azure_node_pool.max_pods_constraint.max_pods_per_node = 1798
    request = gke_multicloud_v1.UpdateAzureNodePoolRequest(azure_node_pool=azure_node_pool)
    operation = client.update_azure_node_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)