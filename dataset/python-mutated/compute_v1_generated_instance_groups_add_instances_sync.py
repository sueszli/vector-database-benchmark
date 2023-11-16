from google.cloud import compute_v1

def sample_add_instances():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstanceGroupsClient()
    request = compute_v1.AddInstancesInstanceGroupRequest(instance_group='instance_group_value', project='project_value', zone='zone_value')
    response = client.add_instances(request=request)
    print(response)