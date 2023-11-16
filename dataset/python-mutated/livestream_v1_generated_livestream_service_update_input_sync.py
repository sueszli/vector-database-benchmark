from google.cloud.video import live_stream_v1

def sample_update_input():
    if False:
        return 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.UpdateInputRequest()
    operation = client.update_input(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)