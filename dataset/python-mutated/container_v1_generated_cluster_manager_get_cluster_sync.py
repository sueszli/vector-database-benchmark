from google.cloud import container_v1

def sample_get_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1.ClusterManagerClient()
    request = container_v1.GetClusterRequest()
    response = client.get_cluster(request=request)
    print(response)