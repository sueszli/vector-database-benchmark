from google.cloud import edgecontainer_v1

def sample_get_cluster():
    if False:
        i = 10
        return i + 15
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.GetClusterRequest(name='name_value')
    response = client.get_cluster(request=request)
    print(response)