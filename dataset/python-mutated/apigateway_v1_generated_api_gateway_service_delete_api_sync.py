from google.cloud import apigateway_v1

def sample_delete_api():
    if False:
        for i in range(10):
            print('nop')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.DeleteApiRequest(name='name_value')
    operation = client.delete_api(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)