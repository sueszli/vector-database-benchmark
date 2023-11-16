from google.cloud import network_services_v1

def sample_update_gateway():
    if False:
        print('Hello World!')
    client = network_services_v1.NetworkServicesClient()
    gateway = network_services_v1.Gateway()
    gateway.name = 'name_value'
    gateway.ports = [569, 570]
    gateway.scope = 'scope_value'
    request = network_services_v1.UpdateGatewayRequest(gateway=gateway)
    operation = client.update_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)