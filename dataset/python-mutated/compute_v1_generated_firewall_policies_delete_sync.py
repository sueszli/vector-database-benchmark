from google.cloud import compute_v1

def sample_delete():
    if False:
        return 10
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.DeleteFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.delete(request=request)
    print(response)