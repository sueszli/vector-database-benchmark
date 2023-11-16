from google.cloud import compute_v1

def sample_set_named_ports():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionInstanceGroupsClient()
    request = compute_v1.SetNamedPortsRegionInstanceGroupRequest(instance_group='instance_group_value', project='project_value', region='region_value')
    response = client.set_named_ports(request=request)
    print(response)