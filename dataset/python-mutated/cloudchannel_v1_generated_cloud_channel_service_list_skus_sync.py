from google.cloud import channel_v1

def sample_list_skus():
    if False:
        return 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListSkusRequest(parent='parent_value', account='account_value')
    page_result = client.list_skus(request=request)
    for response in page_result:
        print(response)