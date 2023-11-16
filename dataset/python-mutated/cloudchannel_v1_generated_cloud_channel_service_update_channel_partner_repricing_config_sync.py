from google.cloud import channel_v1

def sample_update_channel_partner_repricing_config():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    channel_partner_repricing_config = channel_v1.ChannelPartnerRepricingConfig()
    channel_partner_repricing_config.repricing_config.rebilling_basis = 'DIRECT_CUSTOMER_COST'
    request = channel_v1.UpdateChannelPartnerRepricingConfigRequest(channel_partner_repricing_config=channel_partner_repricing_config)
    response = client.update_channel_partner_repricing_config(request=request)
    print(response)