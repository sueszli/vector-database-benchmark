from google.cloud import networkconnectivity_v1alpha1

def sample_get_hub():
    if False:
        return 10
    client = networkconnectivity_v1alpha1.HubServiceClient()
    request = networkconnectivity_v1alpha1.GetHubRequest(name='name_value')
    response = client.get_hub(request=request)
    print(response)