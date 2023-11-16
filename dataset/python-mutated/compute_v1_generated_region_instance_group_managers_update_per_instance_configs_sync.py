from google.cloud import compute_v1

def sample_update_per_instance_configs():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.update_per_instance_configs(request=request)
    print(response)