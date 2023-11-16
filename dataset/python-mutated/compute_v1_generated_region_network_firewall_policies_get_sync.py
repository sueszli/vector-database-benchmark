from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionNetworkFirewallPoliciesClient()
    request = compute_v1.GetRegionNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)