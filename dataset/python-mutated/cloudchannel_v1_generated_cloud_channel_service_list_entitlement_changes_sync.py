from google.cloud import channel_v1

def sample_list_entitlement_changes():
    if False:
        return 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListEntitlementChangesRequest(parent='parent_value')
    page_result = client.list_entitlement_changes(request=request)
    for response in page_result:
        print(response)