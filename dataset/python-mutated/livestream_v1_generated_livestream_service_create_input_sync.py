from google.cloud.video import live_stream_v1

def sample_create_input():
    if False:
        while True:
            i = 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.CreateInputRequest(parent='parent_value', input_id='input_id_value')
    operation = client.create_input(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)