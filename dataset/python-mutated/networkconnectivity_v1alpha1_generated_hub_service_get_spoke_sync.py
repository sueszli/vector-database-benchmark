from google.cloud import networkconnectivity_v1alpha1

def sample_get_spoke():
    if False:
        print('Hello World!')
    client = networkconnectivity_v1alpha1.HubServiceClient()
    request = networkconnectivity_v1alpha1.GetSpokeRequest(name='name_value')
    response = client.get_spoke(request=request)
    print(response)