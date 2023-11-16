from google.cloud import compute_v1

def sample_clone_rules():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.CloneRulesFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.clone_rules(request=request)
    print(response)