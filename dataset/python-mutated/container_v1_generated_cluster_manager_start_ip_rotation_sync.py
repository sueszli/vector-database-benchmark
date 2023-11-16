from google.cloud import container_v1

def sample_start_ip_rotation():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1.ClusterManagerClient()
    request = container_v1.StartIPRotationRequest()
    response = client.start_ip_rotation(request=request)
    print(response)