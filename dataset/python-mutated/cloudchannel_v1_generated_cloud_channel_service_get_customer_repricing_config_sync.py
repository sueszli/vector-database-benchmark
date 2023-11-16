from google.cloud import channel_v1

def sample_get_customer_repricing_config():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.GetCustomerRepricingConfigRequest(name='name_value')
    response = client.get_customer_repricing_config(request=request)
    print(response)