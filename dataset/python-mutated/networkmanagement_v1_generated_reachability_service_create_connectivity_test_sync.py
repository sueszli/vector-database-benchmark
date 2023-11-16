from google.cloud import network_management_v1

def sample_create_connectivity_test():
    if False:
        i = 10
        return i + 15
    client = network_management_v1.ReachabilityServiceClient()
    resource = network_management_v1.ConnectivityTest()
    resource.name = 'name_value'
    request = network_management_v1.CreateConnectivityTestRequest(parent='parent_value', test_id='test_id_value', resource=resource)
    operation = client.create_connectivity_test(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)