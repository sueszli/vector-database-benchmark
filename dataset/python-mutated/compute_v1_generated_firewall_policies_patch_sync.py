from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.PatchFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.patch(request=request)
    print(response)