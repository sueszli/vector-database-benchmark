from google.cloud import networkconnectivity_v1alpha1

def sample_delete_spoke():
    if False:
        i = 10
        return i + 15
    client = networkconnectivity_v1alpha1.HubServiceClient()
    request = networkconnectivity_v1alpha1.DeleteSpokeRequest(name='name_value')
    operation = client.delete_spoke(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)