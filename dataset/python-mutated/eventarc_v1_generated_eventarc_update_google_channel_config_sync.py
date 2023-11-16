from google.cloud import eventarc_v1

def sample_update_google_channel_config():
    if False:
        return 10
    client = eventarc_v1.EventarcClient()
    google_channel_config = eventarc_v1.GoogleChannelConfig()
    google_channel_config.name = 'name_value'
    request = eventarc_v1.UpdateGoogleChannelConfigRequest(google_channel_config=google_channel_config)
    response = client.update_google_channel_config(request=request)
    print(response)