from google.cloud import compute_v1

def sample_list_per_instance_configs():
    if False:
        return 10
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.ListPerInstanceConfigsRegionInstanceGroupManagersRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    page_result = client.list_per_instance_configs(request=request)
    for response in page_result:
        print(response)