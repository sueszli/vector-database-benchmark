from google.cloud import channel_v1

def sample_get_channel_partner_link():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.GetChannelPartnerLinkRequest(name='name_value')
    response = client.get_channel_partner_link(request=request)
    print(response)