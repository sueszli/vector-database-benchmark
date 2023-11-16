from google.cloud import networkconnectivity_v1alpha1

def sample_create_hub():
    if False:
        print('Hello World!')
    client = networkconnectivity_v1alpha1.HubServiceClient()
    request = networkconnectivity_v1alpha1.CreateHubRequest(parent='parent_value')
    operation = client.create_hub(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)