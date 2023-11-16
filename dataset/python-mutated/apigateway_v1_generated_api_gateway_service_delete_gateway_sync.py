from google.cloud import apigateway_v1

def sample_delete_gateway():
    if False:
        for i in range(10):
            print('nop')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.DeleteGatewayRequest(name='name_value')
    operation = client.delete_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)