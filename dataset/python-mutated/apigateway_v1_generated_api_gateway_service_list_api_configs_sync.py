from google.cloud import apigateway_v1

def sample_list_api_configs():
    if False:
        print('Hello World!')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.ListApiConfigsRequest(parent='parent_value')
    page_result = client.list_api_configs(request=request)
    for response in page_result:
        print(response)