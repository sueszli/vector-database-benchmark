from google.cloud import container_v1

def sample_list_node_pools():
    if False:
        while True:
            i = 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.ListNodePoolsRequest()
    response = client.list_node_pools(request=request)
    print(response)