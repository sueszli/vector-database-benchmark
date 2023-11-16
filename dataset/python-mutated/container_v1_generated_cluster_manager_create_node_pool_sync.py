from google.cloud import container_v1

def sample_create_node_pool():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.CreateNodePoolRequest()
    response = client.create_node_pool(request=request)
    print(response)