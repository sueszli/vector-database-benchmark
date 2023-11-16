from google.cloud import apigateway_v1

def sample_create_api_config():
    if False:
        print('Hello World!')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.CreateApiConfigRequest(parent='parent_value', api_config_id='api_config_id_value')
    operation = client.create_api_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)