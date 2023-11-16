from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.GlobalNetworkEndpointGroupsClient()
    request = compute_v1.DeleteGlobalNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value')
    response = client.delete(request=request)
    print(response)