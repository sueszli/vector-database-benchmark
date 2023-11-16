from google.cloud import compute_v1

def sample_list_instances():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstanceGroupsClient()
    request = compute_v1.ListInstancesInstanceGroupsRequest(instance_group='instance_group_value', project='project_value', zone='zone_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)