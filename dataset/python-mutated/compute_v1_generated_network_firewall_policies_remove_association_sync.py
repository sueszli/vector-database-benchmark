from google.cloud import compute_v1

def sample_remove_association():
    if False:
        i = 10
        return i + 15
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.RemoveAssociationNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value')
    response = client.remove_association(request=request)
    print(response)