from google.cloud import networkconnectivity_v1

def sample_get_route():
    if False:
        return 10
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.GetRouteRequest(name='name_value')
    response = client.get_route(request=request)
    print(response)