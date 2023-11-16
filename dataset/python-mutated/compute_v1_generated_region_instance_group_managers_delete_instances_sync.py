from google.cloud import compute_v1

def sample_delete_instances():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.DeleteInstancesRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.delete_instances(request=request)
    print(response)