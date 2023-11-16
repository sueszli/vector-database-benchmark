from google.cloud import compute_v1

def sample_update_peering():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworksClient()
    request = compute_v1.UpdatePeeringNetworkRequest(network='network_value', project='project_value')
    response = client.update_peering(request=request)
    print(response)