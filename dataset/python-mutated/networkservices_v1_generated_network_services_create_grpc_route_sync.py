from google.cloud import network_services_v1

def sample_create_grpc_route():
    if False:
        print('Hello World!')
    client = network_services_v1.NetworkServicesClient()
    grpc_route = network_services_v1.GrpcRoute()
    grpc_route.name = 'name_value'
    grpc_route.hostnames = ['hostnames_value1', 'hostnames_value2']
    request = network_services_v1.CreateGrpcRouteRequest(parent='parent_value', grpc_route_id='grpc_route_id_value', grpc_route=grpc_route)
    operation = client.create_grpc_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)