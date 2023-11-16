from google.cloud import container_v1

def sample_update_cluster():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.UpdateClusterRequest()
    response = client.update_cluster(request=request)
    print(response)