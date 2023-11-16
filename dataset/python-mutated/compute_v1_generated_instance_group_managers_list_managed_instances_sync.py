from google.cloud import compute_v1

def sample_list_managed_instances():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.ListManagedInstancesInstanceGroupManagersRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    page_result = client.list_managed_instances(request=request)
    for response in page_result:
        print(response)