from google.cloud import compute_v1

def sample_add_rule():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionNetworkFirewallPoliciesClient()
    request = compute_v1.AddRuleRegionNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value', region='region_value')
    response = client.add_rule(request=request)
    print(response)