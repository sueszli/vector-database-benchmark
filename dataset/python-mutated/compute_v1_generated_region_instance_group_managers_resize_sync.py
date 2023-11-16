from google.cloud import compute_v1

def sample_resize():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.ResizeRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value', size=443)
    response = client.resize(request=request)
    print(response)