from google.cloud import network_management_v1

def sample_list_connectivity_tests():
    if False:
        print('Hello World!')
    client = network_management_v1.ReachabilityServiceClient()
    request = network_management_v1.ListConnectivityTestsRequest(parent='parent_value')
    page_result = client.list_connectivity_tests(request=request)
    for response in page_result:
        print(response)