from google.cloud import networkconnectivity_v1

def sample_delete_policy_based_route():
    if False:
        print('Hello World!')
    client = networkconnectivity_v1.PolicyBasedRoutingServiceClient()
    request = networkconnectivity_v1.DeletePolicyBasedRouteRequest(name='name_value')
    operation = client.delete_policy_based_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)