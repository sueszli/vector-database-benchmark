from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionDisksClient()
    request = compute_v1.TestIamPermissionsRegionDiskRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.test_iam_permissions(request=request)
    print(response)