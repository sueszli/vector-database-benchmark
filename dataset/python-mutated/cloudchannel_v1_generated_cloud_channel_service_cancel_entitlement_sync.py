from google.cloud import channel_v1

def sample_cancel_entitlement():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.CancelEntitlementRequest(name='name_value')
    operation = client.cancel_entitlement(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)