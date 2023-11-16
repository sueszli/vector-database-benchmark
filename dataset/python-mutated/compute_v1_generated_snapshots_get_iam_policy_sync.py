from google.cloud import compute_v1

def sample_get_iam_policy():
    if False:
        return 10
    client = compute_v1.SnapshotsClient()
    request = compute_v1.GetIamPolicySnapshotRequest(project='project_value', resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)