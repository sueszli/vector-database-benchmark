from google.cloud import container_v1

def sample_delete_node_pool():
    if False:
        while True:
            i = 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.DeleteNodePoolRequest()
    response = client.delete_node_pool(request=request)
    print(response)