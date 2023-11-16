from google.cloud import compute_v1

def sample_abandon_instances():
    if False:
        print('Hello World!')
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.AbandonInstancesRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.abandon_instances(request=request)
    print(response)