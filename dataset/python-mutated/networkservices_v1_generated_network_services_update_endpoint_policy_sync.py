from google.cloud import network_services_v1

def sample_update_endpoint_policy():
    if False:
        for i in range(10):
            print('nop')
    client = network_services_v1.NetworkServicesClient()
    endpoint_policy = network_services_v1.EndpointPolicy()
    endpoint_policy.name = 'name_value'
    endpoint_policy.type_ = 'GRPC_SERVER'
    request = network_services_v1.UpdateEndpointPolicyRequest(endpoint_policy=endpoint_policy)
    operation = client.update_endpoint_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)