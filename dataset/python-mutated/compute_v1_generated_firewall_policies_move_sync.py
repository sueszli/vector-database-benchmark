from google.cloud import compute_v1

def sample_move():
    if False:
        return 10
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.MoveFirewallPolicyRequest(firewall_policy='firewall_policy_value', parent_id='parent_id_value')
    response = client.move(request=request)
    print(response)