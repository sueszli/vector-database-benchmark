from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.GlobalNetworkEndpointGroupsClient()
    request = compute_v1.GetGlobalNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value')
    response = client.get(request=request)
    print(response)