from google.cloud import apigateway_v1

def sample_delete_api_config():
    if False:
        print('Hello World!')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.DeleteApiConfigRequest(name='name_value')
    operation = client.delete_api_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)