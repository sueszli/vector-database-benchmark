from google.cloud import beyondcorp_clientgateways_v1

def sample_delete_client_gateway():
    if False:
        while True:
            i = 10
    client = beyondcorp_clientgateways_v1.ClientGatewaysServiceClient()
    request = beyondcorp_clientgateways_v1.DeleteClientGatewayRequest(name='name_value')
    operation = client.delete_client_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)