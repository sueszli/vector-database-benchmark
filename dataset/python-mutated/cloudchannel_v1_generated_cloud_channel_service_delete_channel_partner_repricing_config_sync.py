from google.cloud import channel_v1

def sample_delete_channel_partner_repricing_config():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.DeleteChannelPartnerRepricingConfigRequest(name='name_value')
    client.delete_channel_partner_repricing_config(request=request)