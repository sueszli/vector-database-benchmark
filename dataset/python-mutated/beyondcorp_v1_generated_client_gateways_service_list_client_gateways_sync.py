from google.cloud import beyondcorp_clientgateways_v1

def sample_list_client_gateways():
    if False:
        while True:
            i = 10
    client = beyondcorp_clientgateways_v1.ClientGatewaysServiceClient()
    request = beyondcorp_clientgateways_v1.ListClientGatewaysRequest(parent='parent_value')
    page_result = client.list_client_gateways(request=request)
    for response in page_result:
        print(response)