from google.cloud import networkconnectivity_v1

def sample_get_policy_based_route():
    if False:
        for i in range(10):
            print('nop')
    client = networkconnectivity_v1.PolicyBasedRoutingServiceClient()
    request = networkconnectivity_v1.GetPolicyBasedRouteRequest(name='name_value')
    response = client.get_policy_based_route(request=request)
    print(response)