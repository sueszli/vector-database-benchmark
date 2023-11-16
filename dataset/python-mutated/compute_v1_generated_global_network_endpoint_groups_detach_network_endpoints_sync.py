from google.cloud import compute_v1

def sample_detach_network_endpoints():
    if False:
        while True:
            i = 10
    client = compute_v1.GlobalNetworkEndpointGroupsClient()
    request = compute_v1.DetachNetworkEndpointsGlobalNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value')
    response = client.detach_network_endpoints(request=request)
    print(response)