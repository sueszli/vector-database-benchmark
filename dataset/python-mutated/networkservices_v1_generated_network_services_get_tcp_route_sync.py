from google.cloud import network_services_v1

def sample_get_tcp_route():
    if False:
        i = 10
        return i + 15
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.GetTcpRouteRequest(name='name_value')
    response = client.get_tcp_route(request=request)
    print(response)