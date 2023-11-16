from google.cloud import compute_v1

def sample_list_instances():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionInstanceGroupsClient()
    request = compute_v1.ListInstancesRegionInstanceGroupsRequest(instance_group='instance_group_value', project='project_value', region='region_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)