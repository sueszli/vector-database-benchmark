from google.cloud import network_management_v1

def sample_get_connectivity_test():
    if False:
        print('Hello World!')
    client = network_management_v1.ReachabilityServiceClient()
    request = network_management_v1.GetConnectivityTestRequest(name='name_value')
    response = client.get_connectivity_test(request=request)
    print(response)