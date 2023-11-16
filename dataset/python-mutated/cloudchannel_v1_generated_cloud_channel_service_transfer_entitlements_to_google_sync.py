from google.cloud import channel_v1

def sample_transfer_entitlements_to_google():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelServiceClient()
    entitlements = channel_v1.Entitlement()
    entitlements.offer = 'offer_value'
    request = channel_v1.TransferEntitlementsToGoogleRequest(parent='parent_value', entitlements=entitlements)
    operation = client.transfer_entitlements_to_google(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)