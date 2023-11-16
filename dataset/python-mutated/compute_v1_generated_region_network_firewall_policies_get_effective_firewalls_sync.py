from google.cloud import compute_v1

def sample_get_effective_firewalls():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionNetworkFirewallPoliciesClient()
    request = compute_v1.GetEffectiveFirewallsRegionNetworkFirewallPolicyRequest(network='network_value', project='project_value', region='region_value')
    response = client.get_effective_firewalls(request=request)
    print(response)