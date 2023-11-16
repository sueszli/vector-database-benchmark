from google.cloud import channel_v1

def sample_list_entitlements():
    if False:
        return 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListEntitlementsRequest(parent='parent_value')
    page_result = client.list_entitlements(request=request)
    for response in page_result:
        print(response)