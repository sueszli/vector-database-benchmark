from google.cloud import eventarc_v1

def sample_get_channel_connection():
    if False:
        i = 10
        return i + 15
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.GetChannelConnectionRequest(name='name_value')
    response = client.get_channel_connection(request=request)
    print(response)