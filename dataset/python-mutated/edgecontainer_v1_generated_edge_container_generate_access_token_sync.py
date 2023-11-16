from google.cloud import edgecontainer_v1

def sample_generate_access_token():
    if False:
        return 10
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.GenerateAccessTokenRequest(cluster='cluster_value')
    response = client.generate_access_token(request=request)
    print(response)