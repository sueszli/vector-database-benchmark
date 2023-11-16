from google.cloud import eventarc_v1

def sample_delete_channel():
    if False:
        while True:
            i = 10
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.DeleteChannelRequest(name='name_value', validate_only=True)
    operation = client.delete_channel(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)