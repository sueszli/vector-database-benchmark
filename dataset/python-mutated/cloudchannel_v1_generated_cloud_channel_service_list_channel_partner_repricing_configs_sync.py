from google.cloud import channel_v1

def sample_list_channel_partner_repricing_configs():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListChannelPartnerRepricingConfigsRequest(parent='parent_value')
    page_result = client.list_channel_partner_repricing_configs(request=request)
    for response in page_result:
        print(response)