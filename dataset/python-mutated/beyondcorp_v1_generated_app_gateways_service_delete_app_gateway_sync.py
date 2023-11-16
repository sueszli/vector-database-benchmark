from google.cloud import beyondcorp_appgateways_v1

def sample_delete_app_gateway():
    if False:
        while True:
            i = 10
    client = beyondcorp_appgateways_v1.AppGatewaysServiceClient()
    request = beyondcorp_appgateways_v1.DeleteAppGatewayRequest(name='name_value')
    operation = client.delete_app_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)