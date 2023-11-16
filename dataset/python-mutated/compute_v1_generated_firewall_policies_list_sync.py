from google.cloud import compute_v1

def sample_list():
    if False:
        print('Hello World!')
    client = compute_v1.FirewallPoliciesClient()
    request = compute_v1.ListFirewallPoliciesRequest()
    page_result = client.list(request=request)
    for response in page_result:
        print(response)