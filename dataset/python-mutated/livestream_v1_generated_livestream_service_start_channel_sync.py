from google.cloud.video import live_stream_v1

def sample_start_channel():
    if False:
        for i in range(10):
            print('nop')
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.StartChannelRequest(name='name_value')
    operation = client.start_channel(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)