from google.cloud import compute_v1

def sample_clone_rules():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionNetworkFirewallPoliciesClient()
    request = compute_v1.CloneRulesRegionNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value', region='region_value')
    response = client.clone_rules(request=request)
    print(response)