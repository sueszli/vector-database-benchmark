from google.cloud import network_services_v1

def sample_delete_endpoint_policy():
    if False:
        i = 10
        return i + 15
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.DeleteEndpointPolicyRequest(name='name_value')
    operation = client.delete_endpoint_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)