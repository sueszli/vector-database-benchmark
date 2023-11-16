from google.cloud.video import live_stream_v1

def sample_stop_channel():
    if False:
        print('Hello World!')
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.StopChannelRequest(name='name_value')
    operation = client.stop_channel(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)