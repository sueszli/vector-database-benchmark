from google.cloud import compute_v1

def sample_get_association():
    if False:
        print('Hello World!')
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.GetAssociationNetworkFirewallPolicyRequest(firewall_policy='firewall_policy_value', project='project_value')
    response = client.get_association(request=request)
    print(response)