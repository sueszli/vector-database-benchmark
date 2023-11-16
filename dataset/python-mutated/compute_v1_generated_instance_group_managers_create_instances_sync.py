from google.cloud import compute_v1

def sample_create_instances():
    if False:
        return 10
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.CreateInstancesInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.create_instances(request=request)
    print(response)