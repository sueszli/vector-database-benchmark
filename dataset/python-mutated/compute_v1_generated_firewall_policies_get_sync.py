from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.GetFirewallPolicyRequest(firewall_policy='firewall_policy_value')
    response = client.get(request=request)
    print(response)