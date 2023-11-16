from google.cloud import container_v1

def sample_delete_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = container_v1.ClusterManagerClient()
    request = container_v1.DeleteClusterRequest()
    response = client.delete_cluster(request=request)
    print(response)