from google.cloud import compute_v1

def sample_apply_updates_to_instances():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.apply_updates_to_instances(request=request)
    print(response)