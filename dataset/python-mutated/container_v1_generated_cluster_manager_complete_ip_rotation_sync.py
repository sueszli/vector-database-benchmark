from google.cloud import container_v1

def sample_complete_ip_rotation():
    if False:
        return 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.CompleteIPRotationRequest()
    response = client.complete_ip_rotation(request=request)
    print(response)