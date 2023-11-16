from google.cloud import network_services_v1

def sample_create_http_route():
    if False:
        i = 10
        return i + 15
    client = network_services_v1.NetworkServicesClient()
    http_route = network_services_v1.HttpRoute()
    http_route.name = 'name_value'
    http_route.hostnames = ['hostnames_value1', 'hostnames_value2']
    request = network_services_v1.CreateHttpRouteRequest(parent='parent_value', http_route_id='http_route_id_value', http_route=http_route)
    operation = client.create_http_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)