from google.cloud import eventarc_v1

def sample_delete_channel_connection():
    if False:
        return 10
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.DeleteChannelConnectionRequest(name='name_value')
    operation = client.delete_channel_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)