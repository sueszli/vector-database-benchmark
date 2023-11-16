from google.cloud import compute_v1

def sample_patch():
    if False:
        print('Hello World!')
    client = compute_v1.NetworksClient()
    request = compute_v1.PatchNetworkRequest(network='network_value', project='project_value')
    response = client.patch(request=request)
    print(response)