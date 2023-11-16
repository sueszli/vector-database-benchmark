from google.cloud import channel_v1

def sample_list_sku_groups():
    if False:
        return 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListSkuGroupsRequest(parent='parent_value')
    page_result = client.list_sku_groups(request=request)
    for response in page_result:
        print(response)