from google.cloud import container_v1

def sample_set_node_pool_management():
    if False:
        return 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetNodePoolManagementRequest()
    response = client.set_node_pool_management(request=request)
    print(response)