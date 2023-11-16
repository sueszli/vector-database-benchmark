from google.cloud import network_services_v1

def sample_get_http_route():
    if False:
        while True:
            i = 10
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.GetHttpRouteRequest(name='name_value')
    response = client.get_http_route(request=request)
    print(response)