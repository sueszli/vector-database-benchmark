from google.cloud import compute_v1

def sample_set_named_ports():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstanceGroupsClient()
    request = compute_v1.SetNamedPortsInstanceGroupRequest(instance_group='instance_group_value', project='project_value', zone='zone_value')
    response = client.set_named_ports(request=request)
    print(response)