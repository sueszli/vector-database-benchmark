from google.cloud import container_v1

def sample_get_node_pool():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1.ClusterManagerClient()
    request = container_v1.GetNodePoolRequest()
    response = client.get_node_pool(request=request)
    print(response)