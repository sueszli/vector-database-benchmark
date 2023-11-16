from google.cloud import network_services_v1

def sample_delete_tcp_route():
    if False:
        print('Hello World!')
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.DeleteTcpRouteRequest(name='name_value')
    operation = client.delete_tcp_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)