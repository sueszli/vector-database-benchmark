from google.cloud import network_services_v1

def sample_list_endpoint_policies():
    if False:
        return 10
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.ListEndpointPoliciesRequest(parent='parent_value')
    page_result = client.list_endpoint_policies(request=request)
    for response in page_result:
        print(response)