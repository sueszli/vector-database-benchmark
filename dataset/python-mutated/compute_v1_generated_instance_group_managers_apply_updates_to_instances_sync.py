from google.cloud import compute_v1

def sample_apply_updates_to_instances():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.ApplyUpdatesToInstancesInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.apply_updates_to_instances(request=request)
    print(response)