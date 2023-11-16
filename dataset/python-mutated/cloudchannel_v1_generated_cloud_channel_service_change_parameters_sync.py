from google.cloud import channel_v1

def sample_change_parameters():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ChangeParametersRequest(name='name_value')
    operation = client.change_parameters(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)