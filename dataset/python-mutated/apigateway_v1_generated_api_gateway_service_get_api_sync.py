from google.cloud import apigateway_v1

def sample_get_api():
    if False:
        print('Hello World!')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.GetApiRequest(name='name_value')
    response = client.get_api(request=request)
    print(response)