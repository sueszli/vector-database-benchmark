from google.cloud import beyondcorp_clientgateways_v1

def sample_get_client_gateway():
    if False:
        print('Hello World!')
    client = beyondcorp_clientgateways_v1.ClientGatewaysServiceClient()
    request = beyondcorp_clientgateways_v1.GetClientGatewayRequest(name='name_value')
    response = client.get_client_gateway(request=request)
    print(response)