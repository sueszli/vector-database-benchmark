from google.cloud import compute_v1

def sample_remove_rule():
    if False:
        print('Hello World!')
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.RemoveRuleNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value')
    response = client.remove_rule(request=request)
    print(response)