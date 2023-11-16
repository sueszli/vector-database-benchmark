from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworkEndpointGroupsClient()
    request = compute_v1.DeleteNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value', zone='zone_value')
    response = client.delete(request=request)
    print(response)