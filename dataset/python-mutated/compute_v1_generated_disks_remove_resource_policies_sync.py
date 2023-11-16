from google.cloud import compute_v1

def sample_remove_resource_policies():
    if False:
        i = 10
        return i + 15
    client = compute_v1.DisksClient()
    request = compute_v1.RemoveResourcePoliciesDiskRequest(disk='disk_value', project='project_value', zone='zone_value')
    response = client.remove_resource_policies(request=request)
    print(response)