from google.cloud import compute_v1

def sample_remove_rule():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionNetworkFirewallPoliciesClient()
    request = compute_v1.RemoveRuleRegionNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value', region='region_value')
    response = client.remove_rule(request=request)
    print(response)