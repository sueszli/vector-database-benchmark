from google.cloud import networkconnectivity_v1

def sample_accept_hub_spoke():
    if False:
        i = 10
        return i + 15
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.AcceptHubSpokeRequest(name='name_value', spoke_uri='spoke_uri_value')
    operation = client.accept_hub_spoke(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)