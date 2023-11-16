from google.cloud import channel_v1

def sample_get_entitlement():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.GetEntitlementRequest(name='name_value')
    response = client.get_entitlement(request=request)
    print(response)