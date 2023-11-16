from google.cloud import network_services_v1

def sample_get_endpoint_policy():
    if False:
        i = 10
        return i + 15
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.GetEndpointPolicyRequest(name='name_value')
    response = client.get_endpoint_policy(request=request)
    print(response)