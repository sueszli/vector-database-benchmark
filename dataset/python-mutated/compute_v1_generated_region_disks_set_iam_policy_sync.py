from google.cloud import compute_v1

def sample_set_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionDisksClient()
    request = compute_v1.SetIamPolicyRegionDiskRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)