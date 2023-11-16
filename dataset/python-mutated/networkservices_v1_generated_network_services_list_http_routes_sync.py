from google.cloud import network_services_v1

def sample_list_http_routes():
    if False:
        return 10
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.ListHttpRoutesRequest(parent='parent_value')
    page_result = client.list_http_routes(request=request)
    for response in page_result:
        print(response)