from google.cloud import compute_v1

def sample_add_association():
    if False:
        i = 10
        return i + 15
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.AddAssociationFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.add_association(request=request)
    print(response)