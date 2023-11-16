from google.cloud import eventarc_publishing_v1

def sample_publish_events():
    if False:
        for i in range(10):
            print('nop')
    client = eventarc_publishing_v1.PublisherClient()
    request = eventarc_publishing_v1.PublishEventsRequest()
    response = client.publish_events(request=request)
    print(response)