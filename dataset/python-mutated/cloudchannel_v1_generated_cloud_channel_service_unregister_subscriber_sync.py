from google.cloud import channel_v1

def sample_unregister_subscriber():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.UnregisterSubscriberRequest(account='account_value', service_account='service_account_value')
    response = client.unregister_subscriber(request=request)
    print(response)