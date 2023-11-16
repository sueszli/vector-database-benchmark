from google.cloud import compute_v1

def sample_add_peering():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NetworksClient()
    request = compute_v1.AddPeeringNetworkRequest(network='network_value', project='project_value')
    response = client.add_peering(request=request)
    print(response)