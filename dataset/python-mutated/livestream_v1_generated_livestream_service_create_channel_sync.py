from google.cloud.video import live_stream_v1

def sample_create_channel():
    if False:
        while True:
            i = 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.CreateChannelRequest(parent='parent_value', channel_id='channel_id_value')
    operation = client.create_channel(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)