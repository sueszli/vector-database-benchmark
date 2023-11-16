from google.cloud import compute_v1

def sample_set_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.SetIamPolicyNodeGroupRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.set_iam_policy(request=request)
    print(response)