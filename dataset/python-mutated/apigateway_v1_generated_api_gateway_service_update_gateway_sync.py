from google.cloud import apigateway_v1

def sample_update_gateway():
    if False:
        i = 10
        return i + 15
    client = apigateway_v1.ApiGatewayServiceClient()
    gateway = apigateway_v1.Gateway()
    gateway.api_config = 'api_config_value'
    request = apigateway_v1.UpdateGatewayRequest(gateway=gateway)
    operation = client.update_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)