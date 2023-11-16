from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.DeleteNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value')
    response = client.delete(request=request)
    print(response)