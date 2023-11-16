from google.cloud import container_v1beta1

def sample_set_node_pool_size():
    if False:
        return 10
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetNodePoolSizeRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', node_pool_id='node_pool_id_value', node_count=1070)
    response = client.set_node_pool_size(request=request)
    print(response)