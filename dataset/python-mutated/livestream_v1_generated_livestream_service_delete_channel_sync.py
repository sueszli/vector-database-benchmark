from google.cloud.video import live_stream_v1

def sample_delete_channel():
    if False:
        for i in range(10):
            print('nop')
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.DeleteChannelRequest(name='name_value')
    operation = client.delete_channel(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)