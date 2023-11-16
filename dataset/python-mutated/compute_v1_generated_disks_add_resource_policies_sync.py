from google.cloud import compute_v1

def sample_add_resource_policies():
    if False:
        print('Hello World!')
    client = compute_v1.DisksClient()
    request = compute_v1.AddResourcePoliciesDiskRequest(disk='disk_value', project='project_value', zone='zone_value')
    response = client.add_resource_policies(request=request)
    print(response)