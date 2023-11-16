from google.cloud import network_management_v1

def sample_update_connectivity_test():
    if False:
        i = 10
        return i + 15
    client = network_management_v1.ReachabilityServiceClient()
    resource = network_management_v1.ConnectivityTest()
    resource.name = 'name_value'
    request = network_management_v1.UpdateConnectivityTestRequest(resource=resource)
    operation = client.update_connectivity_test(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)