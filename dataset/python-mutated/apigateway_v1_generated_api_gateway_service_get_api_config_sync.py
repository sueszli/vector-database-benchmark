from google.cloud import apigateway_v1

def sample_get_api_config():
    if False:
        return 10
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.GetApiConfigRequest(name='name_value')
    response = client.get_api_config(request=request)
    print(response)