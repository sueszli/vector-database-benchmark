from google.cloud import compute_v1

def sample_remove_association():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.RemoveAssociationFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.remove_association(request=request)
    print(response)