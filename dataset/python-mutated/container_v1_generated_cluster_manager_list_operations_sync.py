from google.cloud import container_v1

def sample_list_operations():
    if False:
        return 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.ListOperationsRequest()
    response = client.list_operations(request=request)
    print(response)