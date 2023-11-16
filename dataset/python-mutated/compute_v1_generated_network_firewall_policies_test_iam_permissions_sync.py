from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.TestIamPermissionsNetworkFirewallPolicyRequest(project='project_value', resource='resource_value')
    response = client.test_iam_permissions(request=request)
    print(response)