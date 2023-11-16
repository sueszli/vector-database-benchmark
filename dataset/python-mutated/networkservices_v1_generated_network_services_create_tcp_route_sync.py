from google.cloud import network_services_v1

def sample_create_tcp_route():
    if False:
        for i in range(10):
            print('nop')
    client = network_services_v1.NetworkServicesClient()
    tcp_route = network_services_v1.TcpRoute()
    tcp_route.name = 'name_value'
    request = network_services_v1.CreateTcpRouteRequest(parent='parent_value', tcp_route_id='tcp_route_id_value', tcp_route=tcp_route)
    operation = client.create_tcp_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)