from google.cloud import compute_v1

def sample_set_target_pools():
    if False:
        print('Hello World!')
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.SetTargetPoolsInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.set_target_pools(request=request)
    print(response)