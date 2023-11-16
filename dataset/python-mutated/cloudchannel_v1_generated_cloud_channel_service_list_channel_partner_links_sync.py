from google.cloud import channel_v1

def sample_list_channel_partner_links():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListChannelPartnerLinksRequest(parent='parent_value')
    page_result = client.list_channel_partner_links(request=request)
    for response in page_result:
        print(response)