from google.cloud import eventarc_publishing_v1

def sample_publish_channel_connection_events():
    if False:
        i = 10
        return i + 15
    client = eventarc_publishing_v1.PublisherClient()
    request = eventarc_publishing_v1.PublishChannelConnectionEventsRequest()
    response = client.publish_channel_connection_events(request=request)
    print(response)