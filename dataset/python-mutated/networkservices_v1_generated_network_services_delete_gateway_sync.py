from google.cloud import network_services_v1

def sample_delete_gateway():
    if False:
        for i in range(10):
            print('nop')
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.DeleteGatewayRequest(name='name_value')
    operation = client.delete_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)