from google.cloud import network_management_v1

def sample_delete_connectivity_test():
    if False:
        i = 10
        return i + 15
    client = network_management_v1.ReachabilityServiceClient()
    request = network_management_v1.DeleteConnectivityTestRequest(name='name_value')
    operation = client.delete_connectivity_test(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)