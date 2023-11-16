from google.cloud import apigateway_v1

def sample_list_gateways():
    if False:
        for i in range(10):
            print('nop')
    client = apigateway_v1.ApiGatewayServiceClient()
    request = apigateway_v1.ListGatewaysRequest(parent='parent_value')
    page_result = client.list_gateways(request=request)
    for response in page_result:
        print(response)