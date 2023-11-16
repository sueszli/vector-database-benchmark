from google.cloud import compute_v1

def sample_delete_instances():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.DeleteInstancesInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.delete_instances(request=request)
    print(response)