from google.cloud import eventarc_v1

def sample_create_channel():
    if False:
        for i in range(10):
            print('nop')
    client = eventarc_v1.EventarcClient()
    channel = eventarc_v1.Channel()
    channel.pubsub_topic = 'pubsub_topic_value'
    channel.name = 'name_value'
    request = eventarc_v1.CreateChannelRequest(parent='parent_value', channel=channel, channel_id='channel_id_value', validate_only=True)
    operation = client.create_channel(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)