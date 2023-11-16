from google.cloud import compute_v1

def sample_delete_per_instance_configs():
    if False:
        return 10
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.delete_per_instance_configs(request=request)
    print(response)