from google.cloud import channel_v1

def sample_list_products():
    if False:
        return 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListProductsRequest(account='account_value')
    page_result = client.list_products(request=request)
    for response in page_result:
        print(response)