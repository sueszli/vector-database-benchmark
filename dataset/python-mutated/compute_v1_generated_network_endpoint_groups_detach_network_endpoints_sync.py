from google.cloud import compute_v1

def sample_detach_network_endpoints():
    if False:
        print('Hello World!')
    client = compute_v1.NetworkEndpointGroupsClient()
    request = compute_v1.DetachNetworkEndpointsNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value', zone='zone_value')
    response = client.detach_network_endpoints(request=request)
    print(response)