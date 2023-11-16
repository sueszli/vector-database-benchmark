from google.cloud import compute_v1

def sample_get_iam_policy():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstancesClient()
    request = compute_v1.GetIamPolicyInstanceRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.get_iam_policy(request=request)
    print(response)