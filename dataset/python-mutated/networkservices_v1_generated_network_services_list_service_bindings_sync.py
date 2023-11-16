from google.cloud import network_services_v1

def sample_list_service_bindings():
    if False:
        return 10
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.ListServiceBindingsRequest(parent='parent_value')
    page_result = client.list_service_bindings(request=request)
    for response in page_result:
        print(response)