from google.cloud import channel_v1

def sample_change_renewal_settings():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ChangeRenewalSettingsRequest(name='name_value')
    operation = client.change_renewal_settings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)