from google.cloud import network_services_v1

def sample_get_tls_route():
    if False:
        print('Hello World!')
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.GetTlsRouteRequest(name='name_value')
    response = client.get_tls_route(request=request)
    print(response)