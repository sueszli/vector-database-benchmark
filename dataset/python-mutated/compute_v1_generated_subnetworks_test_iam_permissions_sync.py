from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SubnetworksClient()
    request = compute_v1.TestIamPermissionsSubnetworkRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.test_iam_permissions(request=request)
    print(response)