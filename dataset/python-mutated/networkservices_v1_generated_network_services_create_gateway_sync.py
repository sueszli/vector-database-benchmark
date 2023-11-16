from google.cloud import network_services_v1

def sample_create_gateway():
    if False:
        for i in range(10):
            print('nop')
    client = network_services_v1.NetworkServicesClient()
    gateway = network_services_v1.Gateway()
    gateway.name = 'name_value'
    gateway.ports = [569, 570]
    gateway.scope = 'scope_value'
    request = network_services_v1.CreateGatewayRequest(parent='parent_value', gateway_id='gateway_id_value', gateway=gateway)
    operation = client.create_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)