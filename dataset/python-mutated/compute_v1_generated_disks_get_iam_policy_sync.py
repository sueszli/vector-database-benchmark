from google.cloud import compute_v1

def sample_get_iam_policy():
    if False:
        print('Hello World!')
    client = compute_v1.DisksClient()
    request = compute_v1.GetIamPolicyDiskRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.get_iam_policy(request=request)
    print(response)