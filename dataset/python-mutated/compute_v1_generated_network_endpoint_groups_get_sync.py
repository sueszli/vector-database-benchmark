from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.NetworkEndpointGroupsClient()
    request = compute_v1.GetNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value', zone='zone_value')
    response = client.get(request=request)
    print(response)