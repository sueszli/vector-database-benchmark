from google.cloud import beyondcorp_clientgateways_v1

def sample_create_client_gateway():
    if False:
        i = 10
        return i + 15
    client = beyondcorp_clientgateways_v1.ClientGatewaysServiceClient()
    client_gateway = beyondcorp_clientgateways_v1.ClientGateway()
    client_gateway.name = 'name_value'
    request = beyondcorp_clientgateways_v1.CreateClientGatewayRequest(parent='parent_value', client_gateway=client_gateway)
    operation = client.create_client_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)