from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        i = 10
        return i + 15
    client = compute_v1.DisksClient()
    request = compute_v1.TestIamPermissionsDiskRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.test_iam_permissions(request=request)
    print(response)