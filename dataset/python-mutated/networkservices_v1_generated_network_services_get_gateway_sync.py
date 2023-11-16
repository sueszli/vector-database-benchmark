from google.cloud import network_services_v1

def sample_get_gateway():
    if False:
        i = 10
        return i + 15
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.GetGatewayRequest(name='name_value')
    response = client.get_gateway(request=request)
    print(response)