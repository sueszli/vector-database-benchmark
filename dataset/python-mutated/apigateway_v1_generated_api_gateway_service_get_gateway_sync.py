from google.cloud import apigateway_v1

def sample_get_gateway():
    if False:
        i = 10
        return i + 15
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.GetGatewayRequest(name='name_value')
    response = client.get_gateway(request=request)
    print(response)