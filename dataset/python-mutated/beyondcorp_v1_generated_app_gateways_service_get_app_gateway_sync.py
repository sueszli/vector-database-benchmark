from google.cloud import beyondcorp_appgateways_v1

def sample_get_app_gateway():
    if False:
        print('Hello World!')
    client = beyondcorp_appgateways_v1.AppGatewaysServiceClient()
    request = beyondcorp_appgateways_v1.GetAppGatewayRequest(name='name_value')
    response = client.get_app_gateway(request=request)
    print(response)