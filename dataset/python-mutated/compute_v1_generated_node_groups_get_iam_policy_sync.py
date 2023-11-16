from google.cloud import compute_v1

def sample_get_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.GetIamPolicyNodeGroupRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.get_iam_policy(request=request)
    print(response)