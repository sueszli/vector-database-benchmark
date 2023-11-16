from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.NetworksClient()
    request = compute_v1.GetNetworkRequest(network='network_value', project='project_value')
    response = client.get(request=request)
    print(response)