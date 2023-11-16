from google.cloud import compute_v1

def sample_insert():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NetworkFirewallPoliciesClient()
    request = compute_v1.InsertNetworkFirewallPolicyRequest(project='project_value')
    response = client.insert(request=request)
    print(response)