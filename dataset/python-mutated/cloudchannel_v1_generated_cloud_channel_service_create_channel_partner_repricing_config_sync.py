from google.cloud import channel_v1

def sample_create_channel_partner_repricing_config():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    channel_partner_repricing_config = channel_v1.ChannelPartnerRepricingConfig()
    channel_partner_repricing_config.repricing_config.rebilling_basis = 'DIRECT_CUSTOMER_COST'
    request = channel_v1.CreateChannelPartnerRepricingConfigRequest(parent='parent_value', channel_partner_repricing_config=channel_partner_repricing_config)
    response = client.create_channel_partner_repricing_config(request=request)
    print(response)