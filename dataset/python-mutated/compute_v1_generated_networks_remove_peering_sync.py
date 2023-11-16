from google.cloud import compute_v1

def sample_remove_peering():
    if False:
        while True:
            i = 10
    client = compute_v1.NetworksClient()
    request = compute_v1.RemovePeeringNetworkRequest(network='network_value', project='project_value')
    response = client.remove_peering(request=request)
    print(response)