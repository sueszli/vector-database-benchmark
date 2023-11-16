from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SnapshotsClient()
    request = compute_v1.TestIamPermissionsSnapshotRequest(project='project_value', resource='resource_value')
    response = client.test_iam_permissions(request=request)
    print(response)