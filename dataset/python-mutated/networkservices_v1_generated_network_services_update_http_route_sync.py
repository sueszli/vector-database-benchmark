from google.cloud import network_services_v1

def sample_update_http_route():
    if False:
        for i in range(10):
            print('nop')
    client = network_services_v1.NetworkServicesClient()
    http_route = network_services_v1.HttpRoute()
    http_route.name = 'name_value'
    http_route.hostnames = ['hostnames_value1', 'hostnames_value2']
    request = network_services_v1.UpdateHttpRouteRequest(http_route=http_route)
    operation = client.update_http_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)