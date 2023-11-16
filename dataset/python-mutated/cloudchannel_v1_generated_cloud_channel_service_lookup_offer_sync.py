from google.cloud import channel_v1

def sample_lookup_offer():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.LookupOfferRequest(entitlement='entitlement_value')
    response = client.lookup_offer(request=request)
    print(response)