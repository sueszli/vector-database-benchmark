from google.cloud import channel_v1

def sample_update_channel_partner_link():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    channel_partner_link = channel_v1.ChannelPartnerLink()
    channel_partner_link.reseller_cloud_identity_id = 'reseller_cloud_identity_id_value'
    channel_partner_link.link_state = 'SUSPENDED'
    request = channel_v1.UpdateChannelPartnerLinkRequest(name='name_value', channel_partner_link=channel_partner_link)
    response = client.update_channel_partner_link(request=request)
    print(response)