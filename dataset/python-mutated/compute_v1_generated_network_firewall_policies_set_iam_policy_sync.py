from google.cloud import compute_v1

def sample_set_iam_policy():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.SetIamPolicyNetworkFirewallPolicyRequest(project='project_value', resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)