from google.cloud import container_v1

def sample_create_cluster():
    if False:
        while True:
            i = 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.CreateClusterRequest()
    response = client.create_cluster(request=request)
    print(response)