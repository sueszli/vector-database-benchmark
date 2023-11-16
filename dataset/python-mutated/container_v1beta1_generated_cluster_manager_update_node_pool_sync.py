from google.cloud import container_v1beta1

def sample_update_node_pool():
    if False:
        print('Hello World!')
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.UpdateNodePoolRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', node_pool_id='node_pool_id_value', node_version='node_version_value', image_type='image_type_value')
    response = client.update_node_pool(request=request)
    print(response)