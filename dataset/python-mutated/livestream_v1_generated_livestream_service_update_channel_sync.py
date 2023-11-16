from google.cloud.video import live_stream_v1

def sample_update_channel():
    if False:
        return 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.UpdateChannelRequest()
    operation = client.update_channel(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)