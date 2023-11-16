from google.cloud import compute_v1

def sample_patch_rule():
    if False:
        i = 10
        return i + 15
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.PatchRuleFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.patch_rule(request=request)
    print(response)