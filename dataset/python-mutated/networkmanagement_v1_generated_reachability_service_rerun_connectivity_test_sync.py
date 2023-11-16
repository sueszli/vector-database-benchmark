from google.cloud import network_management_v1

def sample_rerun_connectivity_test():
    if False:
        i = 10
        return i + 15
    client = network_management_v1.ReachabilityServiceClient()
    request = network_management_v1.RerunConnectivityTestRequest(name='name_value')
    operation = client.rerun_connectivity_test(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)