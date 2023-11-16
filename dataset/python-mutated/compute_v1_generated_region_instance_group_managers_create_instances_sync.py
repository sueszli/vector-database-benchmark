from google.cloud import compute_v1

def sample_create_instances():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.CreateInstancesRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.create_instances(request=request)
    print(response)