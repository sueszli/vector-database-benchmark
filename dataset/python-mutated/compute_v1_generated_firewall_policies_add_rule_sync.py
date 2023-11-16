from google.cloud import compute_v1

def sample_add_rule():
    if False:
        print('Hello World!')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.AddRuleFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.add_rule(request=request)
    print(response)