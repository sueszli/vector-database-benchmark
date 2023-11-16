from google.cloud import networkconnectivity_v1

def sample_create_policy_based_route():
    if False:
        i = 10
        return i + 15
    client = networkconnectivity_v1.PolicyBasedRoutingServiceClient()
    policy_based_route = networkconnectivity_v1.PolicyBasedRoute()
    policy_based_route.next_hop_ilb_ip = 'next_hop_ilb_ip_value'
    policy_based_route.network = 'network_value'
    policy_based_route.filter.protocol_version = 'IPV4'
    request = networkconnectivity_v1.CreatePolicyBasedRouteRequest(parent='parent_value', policy_based_route_id='policy_based_route_id_value', policy_based_route=policy_based_route)
    operation = client.create_policy_based_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)