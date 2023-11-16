from google.cloud import eventarc_v1

def sample_create_channel_connection():
    if False:
        print('Hello World!')
    client = eventarc_v1.EventarcClient()
    channel_connection = eventarc_v1.ChannelConnection()
    channel_connection.name = 'name_value'
    channel_connection.channel = 'channel_value'
    request = eventarc_v1.CreateChannelConnectionRequest(parent='parent_value', channel_connection=channel_connection, channel_connection_id='channel_connection_id_value')
    operation = client.create_channel_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)