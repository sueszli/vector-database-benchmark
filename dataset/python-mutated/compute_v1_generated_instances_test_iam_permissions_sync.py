from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.TestIamPermissionsInstanceRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.test_iam_permissions(request=request)
    print(response)