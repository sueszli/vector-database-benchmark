from google.cloud import compute_v1

def sample_list_network_endpoints():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworkEndpointGroupsClient()
    request = compute_v1.ListNetworkEndpointsNetworkEndpointGroupsRequest(network_endpoint_group='network_endpoint_group_value', project='project_value', zone='zone_value')
    page_result = client.list_network_endpoints(request=request)
    for response in page_result:
        print(response)