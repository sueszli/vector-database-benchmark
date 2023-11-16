from google.cloud import apigateway_v1

def sample_create_api():
    if False:
        print('Hello World!')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.CreateApiRequest(parent='parent_value', api_id='api_id_value')
    operation = client.create_api(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)