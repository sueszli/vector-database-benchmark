from google.cloud import compute_v1

def sample_remove_association():
    if False:
        print('Hello World!')
    client = compute_v1.RegionNetworkFirewallPoliciesClient()
    request = compute_v1.RemoveAssociationRegionNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value', region='region_value')
    response = client.remove_association(request=request)
    print(response)