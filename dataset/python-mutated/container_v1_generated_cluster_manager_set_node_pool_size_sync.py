from google.cloud import container_v1

def sample_set_node_pool_size():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetNodePoolSizeRequest(node_count=1070)
    response = client.set_node_pool_size(request=request)
    print(response)