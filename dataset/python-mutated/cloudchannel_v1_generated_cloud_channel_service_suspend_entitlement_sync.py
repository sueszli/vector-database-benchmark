from google.cloud import channel_v1

def sample_suspend_entitlement():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.SuspendEntitlementRequest(name='name_value')
    operation = client.suspend_entitlement(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)