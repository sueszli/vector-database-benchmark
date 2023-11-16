from google.cloud import network_services_v1

def sample_delete_service_binding():
    if False:
        return 10
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.DeleteServiceBindingRequest(name='name_value')
    operation = client.delete_service_binding(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)