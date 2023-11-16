from google.cloud import network_services_v1

def sample_delete_grpc_route():
    if False:
        return 10
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.DeleteGrpcRouteRequest(name='name_value')
    operation = client.delete_grpc_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)