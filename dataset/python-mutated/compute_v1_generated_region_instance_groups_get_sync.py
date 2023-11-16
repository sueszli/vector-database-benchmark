from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionInstanceGroupsClient()
    request = compute_v1.GetRegionInstanceGroupRequest(instance_group='instance_group_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)