from google.cloud import compute_v1

def sample_get_association():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.GetAssociationFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.get_association(request=request)
    print(response)