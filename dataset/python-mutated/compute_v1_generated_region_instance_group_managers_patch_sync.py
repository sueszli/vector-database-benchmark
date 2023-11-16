from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.PatchRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)