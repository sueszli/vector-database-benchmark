from google.cloud import compute_v1

def sample_insert():
    if False:
        print('Hello World!')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.InsertFirewallPolicyRequest(parent_id='parent_id_value')
    response = client.insert(request=request)
    print(response)