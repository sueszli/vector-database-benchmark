from google.cloud import beyondcorp_appgateways_v1

def sample_list_app_gateways():
    if False:
        i = 10
        return i + 15
    client = beyondcorp_appgateways_v1.AppGatewaysServiceClient()
    request = beyondcorp_appgateways_v1.ListAppGatewaysRequest(parent='parent_value')
    page_result = client.list_app_gateways(request=request)
    for response in page_result:
        print(response)