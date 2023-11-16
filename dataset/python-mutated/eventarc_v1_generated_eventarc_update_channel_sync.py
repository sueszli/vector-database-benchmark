from google.cloud import eventarc_v1

def sample_update_channel():
    if False:
        i = 10
        return i + 15
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.UpdateChannelRequest(validate_only=True)
    operation = client.update_channel(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)