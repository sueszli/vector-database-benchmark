from google.cloud import apigateway_v1

def sample_update_api():
    if False:
        print('Hello World!')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.UpdateApiRequest()
    operation = client.update_api(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)