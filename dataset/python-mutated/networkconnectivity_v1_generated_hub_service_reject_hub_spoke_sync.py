from google.cloud import networkconnectivity_v1

def sample_reject_hub_spoke():
    if False:
        for i in range(10):
            print('nop')
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.RejectHubSpokeRequest(name='name_value', spoke_uri='spoke_uri_value')
    operation = client.reject_hub_spoke(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)