from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworksClient()
    request = compute_v1.DeleteNetworkRequest(network='network_value', project='project_value')
    response = client.delete(request=request)
    print(response)