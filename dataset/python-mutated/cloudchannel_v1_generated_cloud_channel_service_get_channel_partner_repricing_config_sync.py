from google.cloud import channel_v1

def sample_get_channel_partner_repricing_config():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.GetChannelPartnerRepricingConfigRequest(name='name_value')
    response = client.get_channel_partner_repricing_config(request=request)
    print(response)