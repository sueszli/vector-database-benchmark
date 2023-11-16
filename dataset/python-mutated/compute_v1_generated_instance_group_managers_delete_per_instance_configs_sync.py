from google.cloud import compute_v1

def sample_delete_per_instance_configs():
    if False:
        print('Hello World!')
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.DeletePerInstanceConfigsInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.delete_per_instance_configs(request=request)
    print(response)