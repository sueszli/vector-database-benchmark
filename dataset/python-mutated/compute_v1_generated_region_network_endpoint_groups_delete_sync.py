from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.RegionNetworkEndpointGroupsClient()
    request = compute_v1.DeleteRegionNetworkEndpointGroupRequest(network_endpoint_group='network_endpoint_group_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)