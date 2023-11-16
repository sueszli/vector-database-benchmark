from google.cloud import networkconnectivity_v1

def sample_delete_spoke():
    if False:
        print('Hello World!')
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.DeleteSpokeRequest(name='name_value')
    operation = client.delete_spoke(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)