from google.cloud import compute_v1

def sample_set_target_pools():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.SetTargetPoolsRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.set_target_pools(request=request)
    print(response)