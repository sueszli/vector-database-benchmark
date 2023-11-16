from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.RegionNetworkEndpointGroupsClient()
    request = compute_v1.GetRegionNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)