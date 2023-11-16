from google.cloud import networkconnectivity_v1

def sample_get_hub():
    if False:
        print('Hello World!')
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.GetHubRequest(name='name_value')
    response = client.get_hub(request=request)
    print(response)