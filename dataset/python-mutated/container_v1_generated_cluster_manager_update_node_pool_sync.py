from google.cloud import container_v1

def sample_update_node_pool():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.UpdateNodePoolRequest(node_version='node_version_value', image_type='image_type_value')
    response = client.update_node_pool(request=request)
    print(response)