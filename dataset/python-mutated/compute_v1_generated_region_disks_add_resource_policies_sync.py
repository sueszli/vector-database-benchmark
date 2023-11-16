from google.cloud import compute_v1

def sample_add_resource_policies():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionDisksClient()
    request = compute_v1.AddResourcePoliciesRegionDiskRequest(disk='disk_value', project='project_value', region='region_value')
    response = client.add_resource_policies(request=request)
    print(response)