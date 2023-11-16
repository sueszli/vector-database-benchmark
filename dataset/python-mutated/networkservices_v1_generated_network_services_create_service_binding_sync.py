from google.cloud import network_services_v1

def sample_create_service_binding():
    if False:
        print('Hello World!')
    client = network_services_v1.NetworkServicesClient()
    service_binding = network_services_v1.ServiceBinding()
    service_binding.name = 'name_value'
    service_binding.service = 'service_value'
    request = network_services_v1.CreateServiceBindingRequest(parent='parent_value', service_binding_id='service_binding_id_value', service_binding=service_binding)
    operation = client.create_service_binding(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)