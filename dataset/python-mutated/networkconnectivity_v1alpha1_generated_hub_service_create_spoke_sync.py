from google.cloud import networkconnectivity_v1alpha1

def sample_create_spoke():
    if False:
        for i in range(10):
            print('nop')
    client = networkconnectivity_v1alpha1.HubServiceClient()
    request = networkconnectivity_v1alpha1.CreateSpokeRequest(parent='parent_value')
    operation = client.create_spoke(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)