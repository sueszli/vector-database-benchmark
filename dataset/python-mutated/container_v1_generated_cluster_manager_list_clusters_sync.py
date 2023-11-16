from google.cloud import container_v1

def sample_list_clusters():
    if False:
        print('Hello World!')
    client = container_v1.ClusterManagerClient()
    request = container_v1.ListClustersRequest()
    response = client.list_clusters(request=request)
    print(response)