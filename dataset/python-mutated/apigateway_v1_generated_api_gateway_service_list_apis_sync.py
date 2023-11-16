from google.cloud import apigateway_v1

def sample_list_apis():
    if False:
        print('Hello World!')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.ListApisRequest(parent='parent_value')
    page_result = client.list_apis(request=request)
    for response in page_result:
        print(response)