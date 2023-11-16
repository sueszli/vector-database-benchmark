from google.cloud import network_services_v1

def sample_update_grpc_route():
    if False:
        print('Hello World!')
    client = network_services_v1.NetworkServicesClient()
    grpc_route = network_services_v1.GrpcRoute()
    grpc_route.name = 'name_value'
    grpc_route.hostnames = ['hostnames_value1', 'hostnames_value2']
    request = network_services_v1.UpdateGrpcRouteRequest(grpc_route=grpc_route)
    operation = client.update_grpc_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)