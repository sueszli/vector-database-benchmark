from google.cloud import networkconnectivity_v1

def sample_get_group():
    if False:
        i = 10
        return i + 15
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.GetGroupRequest(name='name_value')
    response = client.get_group(request=request)
    print(response)