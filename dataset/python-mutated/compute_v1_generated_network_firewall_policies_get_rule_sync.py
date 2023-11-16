from google.cloud import compute_v1

def sample_get_rule():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.GetRuleNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value')
    response = client.get_rule(request=request)
    print(response)