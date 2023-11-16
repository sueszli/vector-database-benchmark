from google.cloud import container_v1beta1

def sample_get_node_pool():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.GetNodePoolRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', node_pool_id='node_pool_id_value')
    response = client.get_node_pool(request=request)
    print(response)