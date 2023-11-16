from google.cloud import compute_v1

def sample_remove_rule():
    if False:
        return 10
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.RemoveRuleFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.remove_rule(request=request)
    print(response)