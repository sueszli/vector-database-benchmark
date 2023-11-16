from google.cloud import networkconnectivity_v1

def sample_get_route_table():
    if False:
        print('Hello World!')
    client = networkconnectivity_v1.HubServiceClient()
    request = networkconnectivity_v1.GetRouteTableRequest(name='name_value')
    response = client.get_route_table(request=request)
    print(response)