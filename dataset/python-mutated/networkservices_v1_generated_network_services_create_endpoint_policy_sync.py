from google.cloud import network_services_v1

def sample_create_endpoint_policy():
    if False:
        print('Hello World!')
    client = network_services_v1.NetworkServicesClient()
    endpoint_policy = network_services_v1.EndpointPolicy()
    endpoint_policy.name = 'name_value'
    endpoint_policy.type_ = 'GRPC_SERVER'
    request = network_services_v1.CreateEndpointPolicyRequest(parent='parent_value', endpoint_policy_id='endpoint_policy_id_value', endpoint_policy=endpoint_policy)
    operation = client.create_endpoint_policy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)