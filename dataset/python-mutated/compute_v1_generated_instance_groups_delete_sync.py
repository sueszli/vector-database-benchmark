from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.InstanceGroupsClient()
    request = compute_v1.DeleteInstanceGroupRequest(instance_group='instance_group_value', project='project_value', zone='zone_value')
    response = client.delete(request=request)
    print(response)