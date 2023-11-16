from google.cloud import network_services_v1

def sample_get_service_binding():
    if False:
        for i in range(10):
            print('nop')
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.GetServiceBindingRequest(name='name_value')
    response = client.get_service_binding(request=request)
    print(response)