from google.cloud import channel_v1

def sample_register_subscriber():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.RegisterSubscriberRequest(account='account_value', service_account='service_account_value')
    response = client.register_subscriber(request=request)
    print(response)