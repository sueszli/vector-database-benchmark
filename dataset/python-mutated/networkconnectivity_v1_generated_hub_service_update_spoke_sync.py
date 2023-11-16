from google.cloud import networkconnectivity_v1

def sample_update_spoke():
    if False:
        while True:
            i = 10
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.UpdateSpokeRequest()
    operation = client.update_spoke(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)