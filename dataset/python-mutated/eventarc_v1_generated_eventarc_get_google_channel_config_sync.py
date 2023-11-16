from google.cloud import eventarc_v1

def sample_get_google_channel_config():
    if False:
        while True:
            i = 10
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.GetGoogleChannelConfigRequest(name='name_value')
    response = client.get_google_channel_config(request=request)
    print(response)