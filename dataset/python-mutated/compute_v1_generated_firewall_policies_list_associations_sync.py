from google.cloud import compute_v1

def sample_list_associations():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.ListAssociationsFirewallPolicyRequest()
    response = client.list_associations(request=request)
    print(response)