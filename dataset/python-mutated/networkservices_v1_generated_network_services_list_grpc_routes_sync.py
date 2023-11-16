from google.cloud import network_services_v1

def sample_list_grpc_routes():
    if False:
        print('Hello World!')
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.ListGrpcRoutesRequest(parent='parent_value')
    page_result = client.list_grpc_routes(request=request)
    for response in page_result:
        print(response)