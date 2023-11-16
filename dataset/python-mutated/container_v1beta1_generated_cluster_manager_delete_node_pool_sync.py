from google.cloud import container_v1beta1

def sample_delete_node_pool():
    if False:
        while True:
            i = 10
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.DeleteNodePoolRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', node_pool_id='node_pool_id_value')
    response = client.delete_node_pool(request=request)
    print(response)