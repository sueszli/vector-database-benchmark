from google.cloud import compute_v1

def sample_get_rule():
    if False:
        print('Hello World!')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.GetRuleFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.get_rule(request=request)
    print(response)