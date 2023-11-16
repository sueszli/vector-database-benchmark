from google.cloud import networkconnectivity_v1alpha1

def sample_delete_hub():
    if False:
        print('Hello World!')
    client = networkconnectivity_v1alpha1.HubServiceClient()
    request = networkconnectivity_v1alpha1.DeleteHubRequest(name='name_value')
    operation = client.delete_hub(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)