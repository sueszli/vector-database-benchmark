from google.cloud import compute_v1

def sample_recreate_instances():
    if False:
        return 10
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.RecreateInstancesInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.recreate_instances(request=request)
    print(response)