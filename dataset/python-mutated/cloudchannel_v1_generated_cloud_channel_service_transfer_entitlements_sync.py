from google.cloud import channel_v1

def sample_transfer_entitlements():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    entitlements = channel_v1.Entitlement()
    entitlements.offer = 'offer_value'
    request = channel_v1.TransferEntitlementsRequest(parent='parent_value', entitlements=entitlements)
    operation = client.transfer_entitlements(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)