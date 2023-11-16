from google.cloud import apigateway_v1

def sample_update_api_config():
    if False:
        return 10
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.UpdateApiConfigRequest()
    operation = client.update_api_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)