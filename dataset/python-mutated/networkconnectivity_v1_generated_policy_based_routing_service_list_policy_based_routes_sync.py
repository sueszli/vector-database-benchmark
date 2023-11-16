from google.cloud import networkconnectivity_v1

def sample_list_policy_based_routes():
    if False:
        i = 10
        return i + 15
    client = networkconnectivity_v1.PolicyBasedRoutingServiceClient()
    request = networkconnectivity_v1.ListPolicyBasedRoutesRequest(parent='parent_value')
    page_result = client.list_policy_based_routes(request=request)
    for response in page_result:
        print(response)