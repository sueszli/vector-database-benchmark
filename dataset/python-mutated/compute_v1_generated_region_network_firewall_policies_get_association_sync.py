from google.cloud import compute_v1

def sample_get_association():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionNetworkFirewallPoliciesClient()
    request = compute_v1.GetAssociationRegionNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value', region='region_value')
    response = client.get_association(request=request)
    print(response)