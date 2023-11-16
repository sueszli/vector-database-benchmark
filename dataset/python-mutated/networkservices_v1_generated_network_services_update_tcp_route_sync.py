from google.cloud import network_services_v1

def sample_update_tcp_route():
    if False:
        while True:
            i = 10
    client = network_services_v1.NetworkServicesClient()
    tcp_route = network_services_v1.TcpRoute()
    tcp_route.name = 'name_value'
    request = network_services_v1.UpdateTcpRouteRequest(tcp_route=tcp_route)
    operation = client.update_tcp_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)