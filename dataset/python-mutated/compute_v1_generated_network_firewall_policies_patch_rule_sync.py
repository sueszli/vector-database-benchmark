from google.cloud import compute_v1

def sample_patch_rule():
    if False:
        return 10
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.PatchRuleNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value')
    response = client.patch_rule(request=request)
    print(response)