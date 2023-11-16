from google.cloud.video import live_stream_v1

def sample_update_pool():
    if False:
        for i in range(10):
            print('nop')
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.UpdatePoolRequest()
    operation = client.update_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)