from google.cloud import network_services_v1

def sample_list_gateways():
    if False:
        i = 10
        return i + 15
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.ListGatewaysRequest(parent='parent_value')
    page_result = client.list_gateways(request=request)
    for response in page_result:
        print(response)