from google.cloud import eventarc_v1

def sample_get_channel():
    if False:
        i = 10
        return i + 15
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.GetChannelRequest(name='name_value')
    response = client.get_channel(request=request)
    print(response)