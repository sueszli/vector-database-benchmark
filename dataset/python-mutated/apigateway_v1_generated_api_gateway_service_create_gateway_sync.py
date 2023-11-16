from google.cloud import apigateway_v1

def sample_create_gateway():
    if False:
        print('Hello World!')
    client = apigateway_v1.ApiGatewayServiceClient()
    gateway = apigateway_v1.Gateway()
    gateway.api_config = 'api_config_value'
    request = apigateway_v1.CreateGatewayRequest(parent='parent_value', gateway_id='gateway_id_value', gateway=gateway)
    operation = client.create_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)