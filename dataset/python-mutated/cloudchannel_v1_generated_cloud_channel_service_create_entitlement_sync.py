from google.cloud import channel_v1

def sample_create_entitlement():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    entitlement = channel_v1.Entitlement()
    entitlement.offer = 'offer_value'
    request = channel_v1.CreateEntitlementRequest(parent='parent_value', entitlement=entitlement)
    operation = client.create_entitlement(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)