from google.cloud import networkconnectivity_v1

def sample_get_spoke():
    if False:
        i = 10
        return i + 15
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.GetSpokeRequest(name='name_value')
    response = client.get_spoke(request=request)
    print(response)