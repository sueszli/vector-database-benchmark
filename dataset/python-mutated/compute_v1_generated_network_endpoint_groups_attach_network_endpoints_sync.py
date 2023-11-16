from google.cloud import compute_v1

def sample_attach_network_endpoints():
    if False:
        while True:
            i = 10
    client = compute_v1.NetworkEndpointGroupsClient()
    request = compute_v1.AttachNetworkEndpointsNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value', zone='zone_value')
    response = client.attach_network_endpoints(request=request)
    print(response)