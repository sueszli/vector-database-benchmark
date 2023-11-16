from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.DeleteRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)