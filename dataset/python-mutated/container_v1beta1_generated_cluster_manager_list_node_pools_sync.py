from google.cloud import container_v1beta1

def sample_list_node_pools():
    if False:
        return 10
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.ListNodePoolsRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.list_node_pools(request=request)
    print(response)