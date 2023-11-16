from google.cloud import network_services_v1

def sample_update_tls_route():
    if False:
        while True:
            i = 10
    client = network_services_v1.NetworkServicesClient()
    tls_route = network_services_v1.TlsRoute()
    tls_route.name = 'name_value'
    tls_route.rules.action.destinations.service_name = 'service_name_value'
    request = network_services_v1.UpdateTlsRouteRequest(tls_route=tls_route)
    operation = client.update_tls_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)