from google.cloud import compute_v1

def sample_add_rule():
    if False:
        while True:
            i = 10
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.AddRuleNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value')
    response = client.add_rule(request=request)
    print(response)