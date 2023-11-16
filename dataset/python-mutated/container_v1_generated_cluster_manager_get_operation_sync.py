from google.cloud import container_v1

def sample_get_operation():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.GetOperationRequest()
    response = client.get_operation(request=request)
    print(response)